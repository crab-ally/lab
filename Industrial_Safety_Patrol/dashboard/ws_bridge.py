#!/usr/bin/env python3
"""
WebSocket Bridge: ROS2 → WebSocket

ROS2 토픽들을 수신하여 연결된 모든 WebSocket 클라이언트에 JSON으로 브로드캐스트합니다.

Subscribed Topics:
    /ttc_alerts     (std_msgs/msg/String)   TTC 충돌 경보
    /detections_2d  (std_msgs/msg/String)   YOLO 탐지 결과
    /odom           (nav_msgs/msg/Odometry) 로봇 위치
    /tracks_3d      (std_msgs/msg/String)   3D 추적 정보

WebSocket Server: ws://0.0.0.0:8765

Usage:
    # ROS2 환경 소스 후 실행
    python3 /workspace/dashboard/ws_bridge.py
"""

import asyncio
import json
import os
import threading
import time
from typing import Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise RuntimeError(
        'websockets 패키지가 필요합니다: pip install websockets'
    )


# ════════════════════════════════════════════════════════════════════════════
# Global state: ROS2 스레드 ↔ asyncio 스레드 공유
# ════════════════════════════════════════════════════════════════════════════

_clients: Set['WebSocketServerProtocol'] = set()
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None  # asyncio 이벤트 루프 (WebSocket 스레드)


def _broadcast(payload: dict) -> None:
    """ROS2 콜백에서 모든 WebSocket 클라이언트에 메시지를 전송합니다."""
    global _loop, _clients
    if _loop is None:
        return
    message = json.dumps(payload)
    with _clients_lock:
        targets = set(_clients)
    if not targets:
        return

    async def _send_all():
        disconnected = set()
        for ws in targets:
            try:
                await ws.send(message)
            except Exception:
                disconnected.add(ws)
        if disconnected:
            with _clients_lock:
                _clients.difference_update(disconnected)

    asyncio.run_coroutine_threadsafe(_send_all(), _loop)


# ════════════════════════════════════════════════════════════════════════════
# ROS2 Bridge Node
# ════════════════════════════════════════════════════════════════════════════

class WsBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('ws_bridge_node')

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(String, '/ttc_alerts',    self._ttc_cb,        reliable_qos)
        self.create_subscription(String, '/detections_2d', self._detections_cb,  reliable_qos)
        self.create_subscription(String, '/tracks_3d',     self._tracks3d_cb,   reliable_qos)
        self.create_subscription(Odometry, '/odom',        self._odom_cb,       sensor_qos)

        self.get_logger().info('[WsBridge] ROS2 subscriptions active.')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _ttc_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        _broadcast({'topic': 'ttc_alerts', 'data': data, 'ts': time.time()})

    def _detections_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        _broadcast({'topic': 'detections_2d', 'data': data, 'ts': time.time()})

    def _tracks3d_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        _broadcast({'topic': 'tracks_3d', 'data': data, 'ts': time.time()})

    def _odom_cb(self, msg: Odometry) -> None:
        payload = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
        }
        _broadcast({'topic': 'odom', 'data': payload, 'ts': time.time()})


# ════════════════════════════════════════════════════════════════════════════
# WebSocket Server
# ════════════════════════════════════════════════════════════════════════════

async def _ws_handler(websocket: 'WebSocketServerProtocol') -> None:
    """새 WebSocket 클라이언트 연결 처리."""
    with _clients_lock:
        _clients.add(websocket)
    remote = websocket.remote_address
    print(f'[WsBridge] Client connected: {remote}  (total={len(_clients)})')
    try:
        # 연결 인사 메시지
        await websocket.send(json.dumps({
            'topic': 'system',
            'data': {'status': 'connected', 'server': 'WsBridge v1.0'},
            'ts': time.time(),
        }))
        # 클라이언트가 끊길 때까지 대기
        await websocket.wait_closed()
    finally:
        with _clients_lock:
            _clients.discard(websocket)
        print(f'[WsBridge] Client disconnected: {remote}  (total={len(_clients)})')


async def _run_ws_server(host: str, port: int) -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    print(f'[WsBridge] WebSocket server listening on ws://{host}:{port}')
    async with websockets.serve(_ws_handler, host, port):
        await asyncio.Future()  # run forever


def _start_ws_thread(host: str, port: int) -> None:
    """별도 스레드에서 asyncio 이벤트 루프를 실행합니다."""
    asyncio.run(_run_ws_server(host, port))


# ════════════════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ws_host = os.environ.get('WS_HOST', '0.0.0.0')
    ws_port = int(os.environ.get('WS_PORT', 8765))

    # 1. WebSocket 서버를 별도 스레드에서 시작
    ws_thread = threading.Thread(
        target=_start_ws_thread,
        args=(ws_host, ws_port),
        daemon=True,
    )
    ws_thread.start()

    # 2. ROS2 노드 실행
    rclpy.init()
    node = WsBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
