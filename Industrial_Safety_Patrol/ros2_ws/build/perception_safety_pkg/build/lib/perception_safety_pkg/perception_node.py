#!/usr/bin/env python3
"""
Node 1: Perception Node (YOLOv8 + DeepSORT Tracking + PPE Check)

Subscribes:
    - /camera/image_raw (sensor_msgs/msg/Image)

Publishes:
    - /detections_2d (std_msgs/msg/String - JSON Format)
      [Fields: track_id, class_name, confidence, bbox [xmin, ymin, xmax, ymax], ppe_ok, timestamp]
    - /perception/debug_image (sensor_msgs/msg/Image)
"""

import json
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('perception_node')

        # ── Parameter Settings ─────────────────────────────────────────
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('target_classes', [0, 1, 2, 3]) # 0: person, 1: helmet, 2: vest, 3: forklift

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.target_classes = self.get_parameter('target_classes').get_parameter_value().integer_array_value

        self.bridge = CvBridge()

        # ── YOLOv8 모델 초기화 ─────────────────────────────────────────
        self.get_logger().info(f'Loading YOLO Model: {self.model_path}...')
        self.model = YOLO(self.model_path)

        # ── DeepSORT 트래커 초기화 ─────────────────────────────────────
        self.get_logger().info('Initializing DeepSORT Tracker...')
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",  # 경량 딥러닝 피처 추출기 사용
            half=True,
            bgr=True
        )

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Subscriptions & Publishers ─────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            sensor_qos
        )

        self.pub_detections_2d = self.create_publisher(
            String,
            '/detections_2d',
            10
        )

        self.pub_debug_img = self.create_publisher(
            Image,
            '/perception/debug_image',
            10
        )

        self.get_logger().info('Node 1: Perception Node (YOLOv8 + DeepSORT) is ready.')

    def _image_callback(self, msg: Image) -> None:
        """카메라 프레임 수신 시 YOLO 감지 -> DeepSORT 추적 -> 2D BBox & Track_ID 토픽 발행"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Exception: {e}')
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        detections_payload = []

        # 1. YOLOv8 감지 수행
        results = self.model.predict(
            source=cv_img,
            conf=self.conf_thresh,
            classes=list(self.target_classes),
            verbose=False
        )

        # 2. DeepSORT 입력 포맷 변환: ([left, top, w, h], confidence, class_name)
        deepsort_input = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls_id]

                    # DeepSORT 포맷 [left, top, width, height]
                    left = float(xyxy[0])
                    top = float(xyxy[1])
                    w = float(xyxy[2] - xyxy[0])
                    h = float(xyxy[3] - xyxy[1])

                    deepsort_input.append(([left, top, w, h], conf, class_name))

        # 3. DeepSORT 객체 추적 업데이트
        tracks = self.tracker.update_tracks(deepsort_input, frame=cv_img)

        # 4. 추적된 결과 정리 및 패킷 생성
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            class_name = track.get_det_class() if track.get_det_class() else 'object'
                
            # BBox (LTRB: left, top, right, bottom)
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(float, ltrb)
            bbox = [xmin, ymin, xmax, ymax]

            # PPE(안전모/조끼) 착용 여부 간이 검사 (Person 대상)
            ppe_ok = self._check_ppe(cv_img, bbox) if class_name == 'person' else True

            det_item = {
                'track_id': track_id,
                'class_name': class_name,
                'confidence': round(float(track.get_det_conf() or 1.0), 3),
                'bbox': [round(v, 1) for v in bbox], # [xmin, ymin, xmax, ymax]
                'ppe_ok': ppe_ok,
                'stamp': stamp_sec
            }
            detections_payload.append(det_item)

            # 디버그 시각화
            self._draw_bbox(cv_img, bbox, track_id, class_name, ppe_ok)

        # ── 1. /detections_2d 토픽 발행 (Node 2 입력용) ───────────────
        json_msg = String()
        json_msg.data = json.dumps({
            'header': {
                'stamp': stamp_sec,
                'frame_id': msg.header.frame_id
            },
            'detections': detections_payload
        }, ensure_ascii=False)
        self.pub_detections_2d.publish(json_msg)

        # ── 2. Debug Image 발행 ───────────────────────────────────────
        if self.pub_debug_img.get_subscription_count() > 0:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.pub_debug_img.publish(debug_msg)

    def _check_ppe(self, img: np.ndarray, bbox: list[float]) -> bool:
        """작업자(Person) 머리 영역(상단 25%) 안전모 색상 감지"""
        h, w, _ = img.shape
        xmin, ymin, xmax, ymax = map(int, bbox)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        if xmax <= xmin or ymax <= ymin:
            return False

        head_roi = img[ymin:ymin + int((ymax - ymin) * 0.25), xmin:xmax]
        if head_roi.size == 0:
            return False

        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        ratio = (cv2.countNonZero(mask) / (head_roi.shape[0] * head_roi.shape[1] + 1e-5))
        return ratio > 0.10

    def _draw_bbox(self, img: np.ndarray, bbox: list[float], track_id: int, class_name: str, ppe_ok: bool) -> None:
        """디버그 Bounding Box 및 Track ID 시각화"""
        xmin, ymin, xmax, ymax = map(int, bbox)
        color = (0, 255, 0) if ppe_ok else (0, 0, 255)

        label = f"ID:{track_id} {class_name}"
        if class_name == 'person':
            label += " (PPE: OK)" if ppe_ok else " (PPE: NO)"

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, label, (xmin, max(15, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Perception Node Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()