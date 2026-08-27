#!/usr/bin/env python3
"""
Node: Event Logger Node

ROS2 안전 이벤트를 SQLite 또는 PostgreSQL에 기록합니다.

Events topic:
    - /odom

    1. FIRE_DETECTION
       - /fire_tracks_3d
       - /camera/image_raw

    2. PPE_VIOLATION
       - /detections_2d
       - /camera/image_raw

    3. TTC_ALERT
       - /ttc_alerts
       - /camera/image_raw

    4. FALL_DETECTION
       - /fall_alarm
       - /camera/image_raw

DB:
    DB_HOST가 설정되어 있으면 PostgreSQL, 없으면 SQLite 사용.

SQLite 경로: /workspace/data/safety_events.db
Images 경로: /workspace/data/event_images/
"""

import json
import os
import sqlite3
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import psycopg2


class SQLiteBackend:
    """SQLite 기반 이벤트 저장소."""

    DDL = """
    CREATE TABLE IF NOT EXISTS safety_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        epoch       REAL   NOT NULL,
        robot_x     REAL   DEFAULT 0.0,
        robot_y     REAL   DEFAULT 0.0,
        event_type  TEXT    NOT NULL,
        severity    TEXT    NOT NULL DEFAULT 'INFO',
        track_id    INTEGER DEFAULT -1,
        image_path  TEXT    DEFAULT '',
        metadata    TEXT    DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_epoch ON safety_events(epoch);
    CREATE INDEX IF NOT EXISTS idx_severity ON safety_events(severity);
    CREATE INDEX IF NOT EXISTS idx_event_type ON safety_events(event_type);
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def insert(self, row: dict) -> None:
        sql = """
        INSERT INTO safety_events
            (timestamp, epoch, robot_x, robot_y, event_type,
             severity, track_id, image_path, metadata)
        VALUES
            (:timestamp, :epoch, :robot_x, :robot_y, :event_type,
             :severity, :track_id, :image_path, :metadata)
        """
        with self._lock:
            self._conn.execute(sql, row)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f"SQLiteBackend({self._path})"


class PostgreSQLBackend:
    """PostgreSQL / TimescaleDB 기반 이벤트 저장소."""

    DDL = """
    CREATE TABLE IF NOT EXISTS safety_events (
        id          SERIAL PRIMARY KEY,
        timestamp   TIMESTAMPTZ NOT NULL,
        epoch       DOUBLE PRECISION NOT NULL,
        robot_x     DOUBLE PRECISION DEFAULT 0.0,
        robot_y     DOUBLE PRECISION DEFAULT 0.0,
        event_type  TEXT NOT NULL,
        severity    TEXT NOT NULL DEFAULT 'INFO',
        track_id    INTEGER DEFAULT -1,
        image_path  TEXT DEFAULT '',
        metadata    JSONB DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_epoch ON safety_events(epoch);
    CREATE INDEX IF NOT EXISTS idx_severity ON safety_events(severity);
    CREATE INDEX IF NOT EXISTS idx_event_type ON safety_events(event_type);
    """

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str) -> None:
        self._dsn = dict(host=host, port=port, dbname=dbname, user=user, password=password)
        self._lock = threading.Lock()
        self._conn = psycopg2.connect(**self._dsn)
        self._conn.autocommit = False

        with self._conn.cursor() as cur:
            cur.execute(self.DDL)

        self._conn.commit()

    def insert(self, row: dict) -> None:
        sql = """
        INSERT INTO safety_events
            (timestamp, epoch, robot_x, robot_y, event_type,
             severity, track_id, image_path, metadata)
        VALUES
            (%(timestamp)s, %(epoch)s, %(robot_x)s, %(robot_y)s,
             %(event_type)s, %(severity)s, %(track_id)s,
             %(image_path)s, %(metadata)s)
        """

        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, row)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f"PostgreSQLBackend({self._dsn['host']}:{self._dsn['port']}/{self._dsn['dbname']})"


class EventLoggerNode(Node):

    def __init__(self) -> None:
        super().__init__('event_logger_node')

        self.declare_parameter('db_path', '/workspace/data/safety_events.db')
        self.declare_parameter('image_dir', '/workspace/data/event_images')
        self.declare_parameter('image_buffer_sec', 5.0)
        self.declare_parameter('image_match_tolerance_sec', 0.3)

        self._db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self._image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        self._image_buffer_sec = self.get_parameter('image_buffer_sec').get_parameter_value().double_value
        self._image_match_tolerance = self.get_parameter('image_match_tolerance_sec').get_parameter_value().double_value

        Path(self._image_dir).mkdir(parents=True, exist_ok=True)

        self._db = self._init_db()
        self.get_logger().info(f'[EventLogger] DB backend: {self._db}')

        self._bridge = CvBridge()
        self._robot_x = 0.0
        self._robot_y = 0.0

        # 동일한 (track_id, key, event_type) 중복 저장 방지
        self._logged_events: set[tuple[int, str, str]] = set()

        # 모든 이벤트에서 사용하는 원본 RGB 이미지 버퍼
        self._camera_images = deque()

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Fire ──────────────────────────────────────────────────────
        self.create_subscription(String, '/fire_tracks_3d', self._fire_tracks_callback, reliable_qos)

        # ── PPE ───────────────────────────────────────────────────────
        self.create_subscription(String, '/detections_2d', self._detections_callback, reliable_qos)

        # ── TTC ───────────────────────────────────────────────────────
        self.create_subscription(String, '/ttc_alerts', self._ttc_callback, reliable_qos)

        # ── Fall ──────────────────────────────────────────────────────
        self.create_subscription(String, '/fall_alarm', self._fall_alarm_callback, reliable_qos)

        # ── Common Camera ─────────────────────────────────────────────
        # 모든 이벤트 사진은 원본 RGB 이미지를 사용
        self.create_subscription(Image, '/camera/image_raw', self._camera_image_callback, sensor_qos)

        # ── Odom ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_callback, sensor_qos)

        self.get_logger().info('EventLogger Node started.')

    def _init_db(self):
        db_host = os.environ.get('DB_HOST', '')

        if db_host:
            return PostgreSQLBackend(
                host=db_host,
                port=int(os.environ.get('DB_PORT', 5432)),
                dbname=os.environ.get('DB_NAME', 'patrol_db'),
                user=os.environ.get('DB_USER', 'admin'),
                password=os.environ.get('DB_PASS', 'password123')
            )

        return SQLiteBackend(self._db_path)

    def _odom_callback(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _camera_image_callback(self, msg: Image) -> None:
        image = self._convert_image(msg)
        if image is None:
            return

        stamp = self._ros_stamp_to_epoch(msg)
        self._camera_images.append((stamp, image))
        self._cleanup_image_buffer(self._camera_images, stamp)

    def _convert_image(self, msg: Image):
        try:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().debug(f'[EventLogger] Image convert error: {e}')
            return None

    def _cleanup_image_buffer(self, buffer, current_stamp: float) -> None:
        cutoff = current_stamp - self._image_buffer_sec

        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

    # ════════════════════════════════════════════════════════════════
    # Fire
    # ════════════════════════════════════════════════════════════════

    def _fire_tracks_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('[EventLogger] Invalid JSON in /fire_tracks_3d')
            return

        if not isinstance(data, dict):
            return

        fires = data.get('fires', [])

        if not isinstance(fires, list) or not fires:
            return

        event_epoch = self._extract_timestamp(data)
        event_epoch = event_epoch if event_epoch is not None else time.time()
        timestamp = self._epoch_to_iso(event_epoch)

        for fire in fires:
            if not isinstance(fire, dict):
                continue

            track_id = self._safe_int(fire.get('fire_id', -1))

            if track_id < 0:
                continue

            key = f'fire_{track_id}'
            event_type = 'FIRE_DETECTION'

            if not self._should_log(track_id, key, event_type):
                continue

            image_path = self._save_buffered_snapshot(
                self._camera_images,
                'FIRE_DETECTION',
                track_id,
                event_epoch,
            )

            row = self._build_row(
                event_type='FIRE_DETECTION',
                severity='WARNING',
                track_id=track_id,
                image_path=image_path,
                timestamp=timestamp,
                epoch=event_epoch,
                metadata={
                    'fire_id': track_id,
                    'position': fire.get('position'),
                    'position_map': fire.get('position_map'),
                    'distance': fire.get('distance'),
                },
            )

            self._write(row, (track_id, key, event_type))

    # ════════════════════════════════════════════════════════════════
    # PPE
    # ════════════════════════════════════════════════════════════════

    def _detections_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('[EventLogger] Invalid JSON in /detections_2d')
            return

        if not isinstance(data, dict):
            return

        detections = data.get('detections', [])

        if not isinstance(detections, list):
            return

        for det in detections:
            if not isinstance(det, dict):
                continue

            if det.get('ppe_ok', True):
                continue

            track_id = self._safe_int(det.get('track_id', -1))
            key = f'ppe_{track_id}'
            event_type = 'PPE_VIOLATION'

            if not self._should_log(track_id, key, event_type):
                continue

            event_epoch = self._extract_timestamp(det)
            event_epoch = event_epoch if event_epoch is not None else time.time()
            timestamp = self._epoch_to_iso(event_epoch)

            image_path = self._save_buffered_snapshot(
                self._camera_images,
                'PPE_VIOLATION',
                track_id,
                event_epoch,
            )

            row = self._build_row(
                event_type='PPE_VIOLATION',
                severity='WARNING',
                track_id=track_id,
                image_path=image_path,
                timestamp=timestamp,
                epoch=event_epoch,
                metadata={
                    'class_name': det.get('class_name', ''),
                    'confidence': det.get('confidence', 0.0),
                    'bbox': det.get('bbox', [])
                },
            )

            self._write(row, (track_id, key, event_type))

    # ════════════════════════════════════════════════════════════════
    # TTC
    # ════════════════════════════════════════════════════════════════

    def _ttc_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('[EventLogger] Invalid JSON in /ttc_alerts')
            return

        if not isinstance(data, dict):
            return

        risk_level = str(data.get('risk_level', 'SAFE')).upper()

        if risk_level not in ('WARNING', 'EMERGENCY'):
            return

        track_id = self._safe_int(data.get('target_track_id', -1))
        key = f'ttc_{track_id}_{risk_level}'
        event_type = 'TTC_ALERT'

        if not self._should_log(track_id, key, event_type):
            return

        event_epoch = self._extract_timestamp(data)
        event_epoch = event_epoch if event_epoch is not None else time.time()
        timestamp = self._epoch_to_iso(event_epoch)

        # TTC는 WARNING / EMERGENCY와 관계없이 하나의 폴더에 저장
        image_path = self._save_buffered_snapshot(
            self._camera_images,
            'TTC_ALERT',
            track_id,
            event_epoch,
        )

        row = self._build_row(
            event_type='TTC_ALERT',
            severity=risk_level,
            track_id=track_id,
            image_path=image_path,
            timestamp=timestamp,
            epoch=event_epoch,
            metadata={
                'risk_level': risk_level,
                'target_subject': data.get('target_subject', ''),
            },
        )

        self._write(row, (track_id, key, event_type))

    # ════════════════════════════════════════════════════════════════
    # Fall Detection
    # ════════════════════════════════════════════════════════════════

    def _fall_alarm_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn('[EventLogger] Invalid JSON in /fall_alarm')
            return

        if not isinstance(data, dict):
            return

        if not data.get('isfallen', False):
            return

        track_id = self._safe_int(data.get('track_id', -1))
        key = f'fall_{track_id}'
        event_type = 'FALL_DETECTION'

        if not self._should_log(track_id, key, event_type):
            return

        event_epoch = self._extract_timestamp(data)
        event_epoch = event_epoch if event_epoch is not None else time.time()
        timestamp = self._epoch_to_iso(event_epoch)

        image_path = self._save_buffered_snapshot(
            self._camera_images,
            'FALL_DETECTION',
            track_id,
            event_epoch,
        )

        row = self._build_row(
            event_type='FALL_DETECTION',
            severity='WARNING',
            track_id=track_id,
            image_path=image_path,
            timestamp=timestamp,
            epoch=event_epoch,
            metadata={
                'class_name': data.get('class_name', ''),
                'confidence': data.get('confidence', 0.0),
                'isfallen': True,
            },
        )

        self._write(row, (track_id, key, event_type))

    # ════════════════════════════════════════════════════════════════
    # Duplicate Check
    # ════════════════════════════════════════════════════════════════

    def _should_log(self, track_id: int, key: str, event_type: str) -> bool:
        event_key = (track_id, key, event_type)
        return event_key not in self._logged_events

    # ════════════════════════════════════════════════════════════════
    # Image Save
    # ════════════════════════════════════════════════════════════════

    def _save_buffered_snapshot(self, buffer, label: str, track_id: int, event_epoch: float) -> str:
        if not buffer:
            self.get_logger().warn(f'[EventLogger] No image available for {label}.')
            return ''

        closest_image = None
        closest_diff = float('inf')

        for stamp, image in buffer:
            diff = abs(stamp - event_epoch)

            if diff < closest_diff:
                closest_diff = diff
                closest_image = image

        if closest_image is None:
            return ''

        if closest_diff > self._image_match_tolerance:
            self.get_logger().warn(f'[EventLogger] Image timestamp mismatch: {closest_diff:.3f}s for {label}')

        try:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            event_dir = Path(self._image_dir) / label
            event_dir.mkdir(parents=True, exist_ok=True)

            filename = f'{label}_track{track_id}_{ts}.jpg'
            filepath = event_dir / filename

            success = cv2.imwrite(str(filepath), closest_image)

            if not success:
                self.get_logger().warn(f'[EventLogger] Failed to write image: {filepath}')
                return ''

            self.get_logger().info(f'[EventLogger] Image saved | track={track_id} | diff={closest_diff:.3f}s | path={filepath}')

            return str(filepath)

        except Exception as e:
            self.get_logger().warn(f'[EventLogger] Snapshot save failed: {e}')
            return ''

    # ════════════════════════════════════════════════════════════════
    # DB Row
    # ════════════════════════════════════════════════════════════════

    def _build_row(
        self,
        event_type: str,
        severity: str,
        track_id: int,
        image_path: str,
        timestamp: str,
        epoch: float,
        metadata: dict,
    ) -> dict:
        return {
            'timestamp': timestamp,
            'epoch': epoch,
            'robot_x': self._robot_x,
            'robot_y': self._robot_y,
            'event_type': event_type,
            'severity': severity,
            'track_id': track_id,
            'image_path': image_path,
            'metadata': json.dumps(metadata, ensure_ascii=False),
        }

    # ════════════════════════════════════════════════════════════════
    # DB Write
    # ════════════════════════════════════════════════════════════════

    def _write(self, row: dict, event_key: tuple[int, str, str]) -> None:
        try:
            self._db.insert(row)
            self._logged_events.add(event_key)

            self.get_logger().info(
                f"[EventLogger] Logged | "
                f"{row['event_type']} | "
                f"{row['severity']} | "
                f"track={row['track_id']} | "
                f"image={row['image_path']} | "
                f"pos=({row['robot_x']:.2f}, {row['robot_y']:.2f})"
            )

        except Exception as e:
            self.get_logger().error(f'[EventLogger] DB write failed: {e}')

    # ════════════════════════════════════════════════════════════════
    # Timestamp Helpers
    # ════════════════════════════════════════════════════════════════

    def _ros_stamp_to_epoch(self, msg: Image) -> float:
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        return float(sec) + float(nanosec) * 1e-9

    def _extract_timestamp(self, data) -> float | None:
        if not isinstance(data, dict):
            return None

        timestamp = data.get('timestamp')

        if timestamp is not None:
            parsed = self._parse_timestamp(timestamp)

            if parsed is not None:
                return parsed

        stamp = data.get('stamp')

        if stamp is not None:
            parsed = self._parse_timestamp(stamp)

            if parsed is not None:
                return parsed

            if isinstance(stamp, dict):
                sec = stamp.get('sec')
                nanosec = stamp.get('nanosec', stamp.get('nsec', 0))

                if sec is not None:
                    try:
                        return float(sec) + float(nanosec) * 1e-9
                    except (TypeError, ValueError):
                        pass

        header = data.get('header')

        if isinstance(header, dict):
            stamp = header.get('stamp')

            if isinstance(stamp, dict):
                sec = stamp.get('sec')
                nanosec = stamp.get('nanosec', stamp.get('nsec', 0))

                if sec is not None:
                    try:
                        return float(sec) + float(nanosec) * 1e-9
                    except (TypeError, ValueError):
                        pass

        return None

    def _parse_timestamp(self, timestamp) -> float | None:
        if isinstance(timestamp, (int, float)):
            return float(timestamp)

        if not isinstance(timestamp, str):
            return None

        try:
            return float(timestamp)
        except ValueError:
            pass

        try:
            value = timestamp.replace('Z', '+00:00')
            dt = datetime.fromisoformat(value)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt.timestamp()

        except (ValueError, TypeError):
            return None

    def _epoch_to_iso(self, epoch: float) -> str:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    def _safe_int(self, value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    # ════════════════════════════════════════════════════════════════
    # Shutdown
    # ════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EventLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()