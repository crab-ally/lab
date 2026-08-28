#!/usr/bin/env python3
"""
WebSocket Bridge: ROS2 → WebSocket

ROS2 안전 데이터를 수신하여 연결된 모든 WebSocket 클라이언트에 JSON으로 브로드캐스트합니다.

Subscribed Topics:
    - /odom           (nav_msgs/msg/Odometry)   로봇 위치/속도

    1. Fire Detection
      - /fire_tracks_3d (std_msgs/msg/String)   화재 3D 위치

    2. PPE Detection
      - /tracks_3d      (std_msgs/msg/String)   3D 객체 추적 정보

    3. TTC Detection
      - /tracks_3d
      - /ttc_alerts     (std_msgs/msg/String)   TTC 충돌 경보

    4. FallDetection
      - /fall_alarm     (std_msgs/msg/String)   쓰러짐 경보

WebSocket Server:
    ws://0.0.0.0:8765
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
    raise RuntimeError('websockets 패키지가 필요합니다: pip install websockets')


_clients: Set['WebSocketServerProtocol'] = set()
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def _broadcast(payload: dict) -> None:
    global _loop, _clients

    if _loop is None:
        return

    message = json.dumps(payload, ensure_ascii=False)

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


class WsBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('ws_bridge_node')

        # ============================================================
        # QoS
        # ============================================================

        sensor_qos = QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
        reliable_qos = QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)

        # ============================================================
        # Subscribed Topics
        # ============================================================

        # Robot Odometry
        self.create_subscription(Odometry,'/odom',self._odom_cb,sensor_qos)

        # Fire Detection
        self.create_subscription(String,'/fire_tracks_3d',self._fire_tracks3d_cb,reliable_qos)

        # PPE Detection / TTC Detection
        self.create_subscription(String,'/tracks_3d',self._tracks3d_cb,reliable_qos)

        # TTC Detection
        self.create_subscription(String,'/ttc_alerts',self._ttc_cb,reliable_qos)

        # Fall Detection
        self.create_subscription(String,'/fall_alarm',self._fall_alarm_cb,reliable_qos)

        self.get_logger().info('[WsBridge] ROS2 subscriptions active.')
        self.get_logger().info('[WsBridge] Topics: /odom /fire_tracks_3d /tracks_3d /ttc_alerts /fall_alarm')

    # ================================================================
    # String Topic Common Handler
    # ================================================================

    def _string_cb(self,msg: String,topic: str) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError,TypeError):
            data = msg.data

        _broadcast({
            'topic': topic,
            'data': data,
            'ts': time.time()
        })

    # ================================================================
    # Fire Detection
    # ================================================================

    def _fire_tracks3d_cb(self,msg: String) -> None:
        self._string_cb(msg,'fire_tracks_3d')

    # ================================================================
    # PPE Detection / TTC Detection
    # ================================================================

    def _tracks3d_cb(self,msg: String) -> None:
        self._string_cb(msg,'tracks_3d')

    # ================================================================
    # TTC Detection
    # ================================================================

    def _ttc_cb(self,msg: String) -> None:
        self._string_cb(msg,'ttc_alerts')

    # ================================================================
    # Fall Detection
    # ================================================================

    def _fall_alarm_cb(self,msg: String) -> None:
        self._string_cb(msg,'fall_alarm')

    # ================================================================
    # Robot Odometry
    # ================================================================

    def _odom_cb(self,msg: Odometry) -> None:
        data = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'z': float(msg.pose.pose.position.z),
            'vx': float(msg.twist.twist.linear.x),
            'vy': float(msg.twist.twist.linear.y)
        }

        _broadcast({
            'topic': 'odom',
            'data': data,
            'ts': time.time()
        })


# ====================================================================
# WebSocket Handler
# ====================================================================

async def _ws_handler(websocket: 'WebSocketServerProtocol') -> None:
    with _clients_lock:
        _clients.add(websocket)
        client_count = len(_clients)

    remote = websocket.remote_address
    print(f'[WsBridge] Client connected: {remote} (total={client_count})')

    try:
        await websocket.send(json.dumps({
            'topic': 'system',
            'data': {
                'status': 'connected',
                'server': 'WsBridge v1.0'
            },
            'ts': time.time()
        },ensure_ascii=False))

        await websocket.wait_closed()

    finally:
        with _clients_lock:
            _clients.discard(websocket)
            client_count = len(_clients)

        print(f'[WsBridge] Client disconnected: {remote} (total={client_count})')


# ====================================================================
# WebSocket Server
# ====================================================================

async def _run_ws_server(host: str,port: int) -> None:
    global _loop

    _loop = asyncio.get_running_loop()

    print(f'[WsBridge] WebSocket server listening on ws://{host}:{port}')

    async with websockets.serve(_ws_handler,host,port):
        await asyncio.Future()


def _start_ws_thread(host: str,port: int) -> None:
    asyncio.run(_run_ws_server(host,port))


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    ws_host = os.environ.get('WS_HOST','0.0.0.0')
    ws_port = int(os.environ.get('WS_PORT','8765'))

    ws_thread = threading.Thread(target=_start_ws_thread,args=(ws_host,ws_port),daemon=True)
    ws_thread.start()

    rclpy.init()
    node = WsBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[WsBridge] stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()