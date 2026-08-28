#!/usr/bin/env python3
"""
WebSocket Bridge: ROS2 + DB → WebSocket

ROS2 실시간 안전 데이터와 EventLoggerNode가 저장한 DB 데이터를 WebSocket으로 Dashboard에 전달합니다.

ROS2:
    /odom
    /fire_tracks_3d
    /tracks_3d
    /ttc_alerts
    /fall_alarm

TF:
    map → odom
    /odom 좌표를 map 좌표로 변환하여 Dashboard에 전달

DB:
    DB_HOST가 있으면 PostgreSQL
    없으면 SQLite

WebSocket:
    ws://0.0.0.0:8765

Client Request:
    {"type":"get_events","limit":100}
    {"type":"get_events","limit":100,"event_type":"FIRE_DETECTION"}
    {"type":"get_events","limit":100,"severity":"WARNING"}
    {"type":"get_event_stats"}
"""

import asyncio
import json
import os
import sqlite3
import threading
import time
from typing import Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point

import websockets
from websockets.server import WebSocketServerProtocol

import psycopg2


_clients: Set['WebSocketServerProtocol'] = set()
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_ws_node = None


# ====================================================================
# WebSocket Broadcast
# ====================================================================

def _broadcast(payload: dict) -> None:
    global _loop, _clients

    if _loop is None:
        return

    message = json.dumps(payload, ensure_ascii=False, default=str)

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


# ====================================================================
# Database Reader
# ====================================================================

class EventDatabaseReader:

    def __init__(self):
        self._lock = threading.Lock()
        self.db_host = os.environ.get('DB_HOST', '')

        if self.db_host:
            self.backend = 'postgresql'
            self.db_path = None
            self.pg_config = {
                'host': self.db_host,
                'port': int(os.environ.get('DB_PORT', 5432)),
                'dbname': os.environ.get('DB_NAME', 'patrol_db'),
                'user': os.environ.get('DB_USER', 'admin'),
                'password': os.environ.get('DB_PASS', 'password123')
            }
            self._conn = psycopg2.connect(**self.pg_config)
        else:
            self.backend = 'sqlite'
            self.db_path = os.environ.get('DB_PATH', '/workspace/data/safety_events.db')
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def _reconnect(self):
        try:
            self._conn.close()
        except Exception:
            pass

        if self.backend == 'postgresql':
            self._conn = psycopg2.connect(**self.pg_config)
        else:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def get_events(self,limit: int = 100,event_type: str | None = None,severity: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 1000))

        sql = """
        SELECT id,timestamp,epoch,robot_x,robot_y,event_type,severity,track_id,image_path,metadata
        FROM safety_events
        WHERE 1=1
        """

        params = []

        if event_type:
            sql += " AND event_type = %s" if self.backend == 'postgresql' else " AND event_type = ?"
            params.append(event_type)

        if severity:
            sql += " AND severity = %s" if self.backend == 'postgresql' else " AND severity = ?"
            params.append(severity)

        sql += " ORDER BY epoch DESC LIMIT %s" if self.backend == 'postgresql' else " ORDER BY epoch DESC LIMIT ?"
        params.append(limit)

        columns = ['id','timestamp','epoch','robot_x','robot_y','event_type','severity','track_id','image_path','metadata']

        with self._lock:
            try:
                if self.backend == 'postgresql':
                    with self._conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                else:
                    cursor = self._conn.cursor()
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()

                result = []

                for row in rows:
                    item = dict(zip(columns, row))
                    item['metadata'] = self._parse_metadata(item['metadata'])
                    result.append(item)

                return result

            except Exception:
                self._reconnect()
                raise

    def get_stats(self) -> dict:
        sql = """
        SELECT event_type,severity,COUNT(*)
        FROM safety_events
        GROUP BY event_type,severity
        ORDER BY event_type,severity
        """

        with self._lock:
            try:
                if self.backend == 'postgresql':
                    with self._conn.cursor() as cur:
                        cur.execute(sql)
                        rows = cur.fetchall()
                else:
                    cursor = self._conn.cursor()
                    cursor.execute(sql)
                    rows = cursor.fetchall()

                stats = {
                    'total': 0,
                    'by_event_type': {},
                    'by_severity': {}
                }

                for event_type,severity,count in rows:
                    count = int(count)
                    stats['total'] += count
                    stats['by_event_type'][event_type] = stats['by_event_type'].get(event_type,0) + count
                    stats['by_severity'][severity] = stats['by_severity'].get(severity,0) + count

                return stats

            except Exception:
                self._reconnect()
                raise

    @staticmethod
    def _parse_metadata(metadata):
        if metadata is None:
            return {}

        if isinstance(metadata,dict):
            return metadata

        if isinstance(metadata,str):
            try:
                value = json.loads(metadata)
                return value if isinstance(value,dict) else {}
            except json.JSONDecodeError:
                return {}

        return {}

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ====================================================================
# WsBridge Node
# ====================================================================

class WsBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('ws_bridge_node')

        self.declare_parameter('history_limit',100)
        self._history_limit = int(self.get_parameter('history_limit').value)

        self._db = EventDatabaseReader()

        self.get_logger().info(f'[WsBridge] DB backend: {self._db.backend}')

        # ============================================================
        # TF
        # ============================================================

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer,self)

        self._map_frame = 'map'
        self._odom_frame = 'odom'

        # ============================================================
        # QoS
        # ============================================================

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # ============================================================
        # ROS2 subscriptions
        # ============================================================

        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_cb,
            sensor_qos
        )

        self.create_subscription(
            String,
            '/fire_tracks_3d',
            self._fire_tracks3d_cb,
            reliable_qos
        )

        self.create_subscription(
            String,
            '/tracks_3d',
            self._tracks3d_cb,
            reliable_qos
        )

        self.create_subscription(
            String,
            '/ttc_alerts',
            self._ttc_cb,
            reliable_qos
        )

        self.create_subscription(
            String,
            '/fall_alarm',
            self._fall_alarm_cb,
            reliable_qos
        )

        self.get_logger().info('[WsBridge] ROS2 subscriptions active.')

    # ================================================================
    # Odom → Map Transform
    # ================================================================

    def _odom_to_map(self,msg: Odometry) -> tuple[float,float,float]:
        point = PointStamped()
        point.header = msg.header
        point.header.frame_id = self._odom_frame
        point.point.x = float(msg.pose.pose.position.x)
        point.point.y = float(msg.pose.pose.position.y)
        point.point.z = float(msg.pose.pose.position.z)

        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._odom_frame,
                rclpy.time.Time()
            )

            transformed = do_transform_point(point,transform)

            return (
                float(transformed.point.x),
                float(transformed.point.y),
                float(transformed.point.z)
            )

        except Exception as e:
            self.get_logger().debug(
                f'[WsBridge] map TF unavailable, using odom coordinates: {e}'
            )

            return (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(msg.pose.pose.position.z)
            )

    # ================================================================
    # ROS String Handler
    # ================================================================

    def _string_cb(self,msg: String,topic: str) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError,TypeError):
            data = msg.data

        _broadcast({
            'type':'realtime',
            'topic':topic,
            'data':data,
            'ts':time.time()
        })

    def _fire_tracks3d_cb(self,msg: String) -> None:
        self._string_cb(msg,'fire_tracks_3d')

    def _tracks3d_cb(self,msg: String) -> None:
        self._string_cb(msg,'tracks_3d')

    def _ttc_cb(self,msg: String) -> None:
        self._string_cb(msg,'ttc_alerts')

    def _fall_alarm_cb(self,msg: String) -> None:
        self._string_cb(msg,'fall_alarm')

    # ================================================================
    # Odom
    # ================================================================

    def _odom_cb(self,msg: Odometry) -> None:
        odom_x = float(msg.pose.pose.position.x)
        odom_y = float(msg.pose.pose.position.y)
        odom_z = float(msg.pose.pose.position.z)

        map_x,map_y,map_z = self._odom_to_map(msg)

        data = {
            'x':odom_x,
            'y':odom_y,
            'z':odom_z,
            'map_x':map_x,
            'map_y':map_y,
            'map_z':map_z,
            'frame_id':'map',
            'source_frame':'odom',
            'vx':float(msg.twist.twist.linear.x),
            'vy':float(msg.twist.twist.linear.y)
        }

        _broadcast({
            'type':'realtime',
            'topic':'odom',
            'data':data,
            'ts':time.time()
        })

    # ================================================================
    # DB History
    # ================================================================

    def get_event_history(self,limit=100,event_type=None,severity=None) -> dict:
        try:
            events = self._db.get_events(limit,event_type,severity)

            return {
                'type':'event_history',
                'data':events,
                'count':len(events),
                'ts':time.time()
            }

        except Exception as e:
            self.get_logger().error(f'[WsBridge] DB event query failed: {e}')

            return {
                'type':'error',
                'error':'event_history_query_failed',
                'message':str(e),
                'ts':time.time()
            }

    # ================================================================
    # DB Statistics
    # ================================================================

    def get_event_stats(self) -> dict:
        try:
            stats = self._db.get_stats()

            return {
                'type':'event_stats',
                'data':stats,
                'ts':time.time()
            }

        except Exception as e:
            self.get_logger().error(f'[WsBridge] DB stats query failed: {e}')

            return {
                'type':'error',
                'error':'event_stats_query_failed',
                'message':str(e),
                'ts':time.time()
            }

    # ================================================================
    # Shutdown
    # ================================================================

    def destroy_node(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

        super().destroy_node()


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
            'type':'system',
            'topic':'system',
            'data':{
                'status':'connected',
                'server':'WsBridge v2.0'
            },
            'ts':time.time()
        },ensure_ascii=False))

        history = await asyncio.to_thread(
            _ws_node.get_event_history,
            _ws_node._history_limit,
            None,
            None
        )

        await websocket.send(json.dumps(history,ensure_ascii=False,default=str))

        stats = await asyncio.to_thread(_ws_node.get_event_stats)

        await websocket.send(json.dumps(stats,ensure_ascii=False,default=str))

        async for message in websocket:
            try:
                request = json.loads(message)
            except (json.JSONDecodeError,TypeError):
                await websocket.send(json.dumps({
                    'type':'error',
                    'error':'invalid_json',
                    'ts':time.time()
                },ensure_ascii=False))
                continue

            if not isinstance(request,dict):
                continue

            request_type = request.get('type','')

            if request_type == 'get_events':
                limit = request.get('limit',_ws_node._history_limit)
                event_type = request.get('event_type')
                severity = request.get('severity')

                result = await asyncio.to_thread(
                    _ws_node.get_event_history,
                    limit,
                    event_type,
                    severity
                )

                await websocket.send(
                    json.dumps(result,ensure_ascii=False,default=str)
                )

            elif request_type == 'get_event_stats':
                result = await asyncio.to_thread(
                    _ws_node.get_event_stats
                )

                await websocket.send(
                    json.dumps(result,ensure_ascii=False,default=str)
                )

            elif request_type == 'ping':
                await websocket.send(json.dumps({
                    'type':'pong',
                    'ts':time.time()
                },ensure_ascii=False))

    except Exception as e:
        print(f'[WsBridge] WebSocket error: {remote}: {e}')

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
    global _ws_node

    ws_host = os.environ.get('WS_HOST','0.0.0.0')
    ws_port = int(os.environ.get('WS_PORT','8765'))

    ws_thread = threading.Thread(
        target=_start_ws_thread,
        args=(ws_host,ws_port),
        daemon=True
    )
    ws_thread.start()

    rclpy.init()
    _ws_node = WsBridgeNode()

    try:
        rclpy.spin(_ws_node)
    except KeyboardInterrupt:
        _ws_node.get_logger().info('WsBridge stopped.')
    finally:
        _ws_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()