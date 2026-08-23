#!/usr/bin/env python3
"""
Node: Event Logger Node

ROS2 이벤트(TTC 경보, YOLO 탐지, 로봇 위치)를 SQLite 또는 PostgreSQL에 기록합니다.

Subscribes:
    - /ttc_alerts     (std_msgs/msg/String - JSON)  TTC 충돌 위험 경보
    - /detections_2d  (std_msgs/msg/String - JSON)  YOLO + DeepSORT 탐지 결과
    - /odom           (nav_msgs/msg/Odometry)        로봇 위치/속도
    - /camera/image_raw (sensor_msgs/msg/Image)      위험 이벤트 스냅샷용

DB 선택:
    환경변수 DB_HOST 가 설정되어 있으면 PostgreSQL, 없으면 SQLite 사용.

SQLite 경로: /workspace/data/safety_events.db
Image 저장:  /workspace/data/event_images/
"""

import json
import os
import sqlite3
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ────────────────────────────────────────────────────────────────────────────
# PostgreSQL: psycopg2 는 선택적 의존성이므로 ImportError를 허용
# ────────────────────────────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# Database Backends
# ════════════════════════════════════════════════════════════════════════════

class SQLiteBackend:
    """SQLite 기반 이벤트 저장소."""

    DDL = """
    CREATE TABLE IF NOT EXISTS safety_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        epoch       REAL    NOT NULL,
        robot_x     REAL    DEFAULT 0.0,
        robot_y     REAL    DEFAULT 0.0,
        event_type  TEXT    NOT NULL,
        severity    TEXT    NOT NULL DEFAULT 'INFO',
        track_id    INTEGER DEFAULT -1,
        image_path  TEXT    DEFAULT '',
        metadata    TEXT    DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_epoch     ON safety_events(epoch);
    CREATE INDEX IF NOT EXISTS idx_severity  ON safety_events(severity);
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
            (timestamp, epoch, robot_x, robot_y, event_type, severity, track_id, image_path, metadata)
        VALUES
            (:timestamp, :epoch, :robot_x, :robot_y, :event_type, :severity, :track_id, :image_path, :metadata)
        """
        with self._lock:
            self._conn.execute(sql, row)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f'SQLiteBackend({self._path})'


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
    CREATE INDEX IF NOT EXISTS idx_epoch      ON safety_events(epoch);
    CREATE INDEX IF NOT EXISTS idx_severity   ON safety_events(severity);
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
            (timestamp, epoch, robot_x, robot_y, event_type, severity, track_id, image_path, metadata)
        VALUES
            (%(timestamp)s, %(epoch)s, %(robot_x)s, %(robot_y)s,
             %(event_type)s, %(severity)s, %(track_id)s, %(image_path)s, %(metadata)s)
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


# ════════════════════════════════════════════════════════════════════════════
# Event Logger Node
# ════════════════════════════════════════════════════════════════════════════

class EventLoggerNode(Node):
    """
    ROS2 이벤트 로거 노드.

    /ttc_alerts, /detections_2d 토픽을 수신하여 DB에 저장합니다.
    위험 이벤트 발생 시 /camera/image_raw 의 최신 프레임을 스냅샷으로 저장합니다.
    """

    def __init__(self) -> None:
        super().__init__('event_logger_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('db_path', '/workspace/data/safety_events.db')
        self.declare_parameter('image_dir', '/workspace/data/event_images')
        self.declare_parameter('save_image_on_warning', True)
        self.declare_parameter('save_image_on_emergency', True)
        self.declare_parameter('min_log_interval_sec', 1.0)   # 동일 track_id 중복 기록 방지

        self._db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self._image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        self._save_img_warn = self.get_parameter('save_image_on_warning').get_parameter_value().bool_value
        self._save_img_emrg = self.get_parameter('save_image_on_emergency').get_parameter_value().bool_value
        self._min_interval = self.get_parameter('min_log_interval_sec').get_parameter_value().double_value

        Path(self._image_dir).mkdir(parents=True, exist_ok=True)

        # ── DB Backend 선택 ─────────────────────────────────────────────────
        self._db = self._init_db()
        self.get_logger().info(f'[EventLogger] DB backend: {self._db}')

        # ── 상태 변수 ───────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._latest_frame: cv2.Mat | None = None       # 최신 카메라 프레임
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._last_log_time: dict[str, float] = {}      # track_id → last epoch

        # ── QoS ─────────────────────────────────────────────────────────────
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

        # ── Subscriptions ────────────────────────────────────────────────────
        self.create_subscription(
            String,
            '/ttc_alerts',
            self._ttc_callback,
            reliable_qos,
        )
        self.create_subscription(
            String,
            '/detections_2d',
            self._detections_callback,
            reliable_qos,
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            sensor_qos,
        )

        self.get_logger().info('[EventLogger] Node started. Listening for safety events...')

    # ── DB Init ──────────────────────────────────────────────────────────────

    def _init_db(self):
        db_host = os.environ.get('DB_HOST', '')
        if db_host and _PSYCOPG2_AVAILABLE:
            return PostgreSQLBackend(
                host=db_host,
                port=int(os.environ.get('DB_PORT', 5432)),
                dbname=os.environ.get('DB_NAME', 'patrol_db'),
                user=os.environ.get('DB_USER', 'admin'),
                password=os.environ.get('DB_PASS', 'password123'),
            )
        else:
            if db_host and not _PSYCOPG2_AVAILABLE:
                self.get_logger().warn('[EventLogger] psycopg2 not available; falling back to SQLite.')
            return SQLiteBackend(self._db_path)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry) -> None:
        """로봇 위치 업데이트."""
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _image_callback(self, msg: Image) -> None:
        """최신 카메라 프레임 버퍼링."""
        try:
            self._latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().debug(f'[EventLogger] Image convert error: {e}')

    def _ttc_callback(self, msg: String) -> None:
        """
        TTC 경보 처리.

        Expected JSON fields:
            min_ttc, risk_level, target_track_id, target_subject, timestamp
        """
        try:
            data: dict = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[EventLogger] Invalid JSON in /ttc_alerts')
            return

        risk_level: str = data.get('risk_level', 'SAFE').upper()
        if risk_level == 'SAFE':
            return  # SAFE 수준은 기록하지 않음

        track_id: int = data.get('target_track_id', -1)
        key = f'ttc_{track_id}'

        if not self._should_log(key):
            return

        severity = 'EMERGENCY' if risk_level == 'EMERGENCY' else 'WARNING'
        image_path = ''

        if (severity == 'EMERGENCY' and self._save_img_emrg) or \
           (severity == 'WARNING' and self._save_img_warn):
            image_path = self._save_snapshot(severity, track_id)

        row = self._build_row(
            event_type='TTC_ALERT',
            severity=severity,
            track_id=track_id,
            image_path=image_path,
            metadata={
                'min_ttc': data.get('min_ttc'),
                'risk_level': risk_level,
                'target_subject': data.get('target_subject', ''),
            },
        )
        self._write(row, key)

    def _detections_callback(self, msg: String) -> None:
        """
        YOLO + DeepSORT 탐지 결과 처리.

        PPE 위반(ppe_ok == False) 이벤트만 기록합니다.
        Expected JSON: list of detection dicts with fields
            track_id, class_name, confidence, bbox, ppe_ok, timestamp
        """
        try:
            detections: list = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(detections, list):
            return

        for det in detections:
            ppe_ok = det.get('ppe_ok', True)
            class_name = det.get('class_name', '')
            track_id = det.get('track_id', -1)

            # PPE 미착용 작업자만 기록
            if class_name != 'person' or ppe_ok:
                continue

            key = f'ppe_{track_id}'
            if not self._should_log(key):
                continue

            image_path = self._save_snapshot('PPE_VIOLATION', track_id) if self._save_img_warn else ''

            row = self._build_row(
                event_type='PPE_VIOLATION',
                severity='WARNING',
                track_id=track_id,
                image_path=image_path,
                metadata={
                    'class_name': class_name,
                    'confidence': det.get('confidence', 0.0),
                    'bbox': det.get('bbox', []),
                },
            )
            self._write(row, key)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _should_log(self, key: str) -> bool:
        """동일 키에 대해 min_log_interval_sec 이내 중복 기록을 방지합니다."""
        now = time.time()
        last = self._last_log_time.get(key, 0.0)
        if (now - last) < self._min_interval:
            return False
        self._last_log_time[key] = now
        return True

    def _save_snapshot(self, label: str, track_id: int) -> str:
        """현재 카메라 프레임을 이미지 파일로 저장하고 경로를 반환합니다."""
        if self._latest_frame is None:
            return ''
        try:
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{label}_track{track_id}_{ts}.jpg'
            filepath = os.path.join(self._image_dir, filename)
            cv2.imwrite(filepath, self._latest_frame)
            return filepath
        except Exception as e:
            self.get_logger().warn(f'[EventLogger] Snapshot save failed: {e}')
            return ''

    def _build_row(
        self,
        event_type: str,
        severity: str,
        track_id: int,
        image_path: str,
        metadata: dict,
    ) -> dict:
        now = datetime.utcnow()
        return {
            'timestamp': now.isoformat(),
            'epoch': now.timestamp(),
            'robot_x': self._robot_x,
            'robot_y': self._robot_y,
            'event_type': event_type,
            'severity': severity,
            'track_id': track_id,
            'image_path': image_path,
            'metadata': json.dumps(metadata),
        }

    def _write(self, row: dict, key: str) -> None:
        try:
            self._db.insert(row)
            self.get_logger().info(
                f"[EventLogger] ✅ Logged | {row['event_type']} | {row['severity']} "
                f"| track={row['track_id']} | pos=({row['robot_x']:.2f}, {row['robot_y']:.2f})"
            )
        except Exception as e:
            self.get_logger().error(f'[EventLogger] DB write failed: {e}')

    def destroy_node(self) -> None:
        self._db.close()
        super().destroy_node()


# ════════════════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════════════════

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
