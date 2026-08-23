#!/usr/bin/env python3
"""
Node 1: Fire Detection Node (HSV Color Segmentation + 2D Temporal BBox Tracking)

Subscribes:
    - /camera/image_raw (sensor_msgs/msg/Image)
    - /camera/depth/image_raw (sensor_msgs/msg/Image)

Publishes:
    - /fire_candidates (std_msgs/msg/String - JSON Format)
"""

import json
import math
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters


class FireDetectionNode(Node):
    """HSV 기반 화재 2D BBox 감지 및 시계열 트래킹 노드"""

    def __init__(self) -> None:
        super().__init__('fire_detection_node')

        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('confirm_frames', 10)
        self.declare_parameter('max_missed_frames', 1)
        self.declare_parameter('bbox_iou_threshold', 0.25)
        self.declare_parameter('center_distance_threshold', 80.0)

        self.confirm_frames = self.get_parameter('confirm_frames').get_parameter_value().integer_value
        self.max_missed_frames = self.get_parameter('max_missed_frames').get_parameter_value().integer_value
        self.bbox_iou_threshold = self.get_parameter('bbox_iou_threshold').get_parameter_value().double_value
        self.center_distance_threshold = self.get_parameter('center_distance_threshold').get_parameter_value().double_value

        # 시계열 BBox 트래킹 저장소
        self.temporal_tracks: Dict[int, dict] = {}
        self.next_track_id: int = 0

        # ── QoS Profile ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Message Filters Synchronizer ──────────────────────────────
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw', qos_profile=sensor_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=sensor_qos
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # ── Publisher ─────────────────────────────────────────────────
        self.candidate_pub = self.create_publisher(
            String,
            '/fire_candidates',
            10
        )

        self.get_logger().info(
            f'Fire Detection Node started. confirmation_frames={self.confirm_frames}'
        )

    @staticmethod
    def bbox_iou(a: List[int], b: List[int]) -> float:
        """2D Bounding Box IoU 계산 ([xmin, ymin, xmax, ymax])"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return float(inter / union)

    @staticmethod
    def bbox_center(bbox: List[int]) -> Tuple[float, float]:
        """BBox 중심점 좌표 계산"""
        return (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0
        )

    @classmethod
    def center_distance(cls, a: List[int], b: List[int]) -> float:
        """두 BBox 중심점 간 유클리드 거리 계산"""
        ax, ay = cls.bbox_center(a)
        bx, by = cls.bbox_center(b)
        return float(math.hypot(ax - bx, ay - by))

    def _match_candidate(self, candidate: dict, matched_ids: Set[int]) -> Optional[int]:
        """기존 트랙과 신규 감지 객체 간 IoU + 거리 기반 매칭"""
        best_id: Optional[int] = None
        best_score: float = -1.0

        bbox = candidate['bbox']

        for track_id, track in self.temporal_tracks.items():
            if track_id in matched_ids:
                continue

            old_bbox = track['bbox']

            iou = self.bbox_iou(bbox, old_bbox)
            distance = self.center_distance(bbox, old_bbox)

            if iou < self.bbox_iou_threshold and distance > self.center_distance_threshold:
                continue

            distance_score = max(0.0, 1.0 - distance / self.center_distance_threshold)
            score = 0.7 * iou + 0.3 * distance_score

            if score > best_score:
                best_score = score
                best_id = track_id

        return best_id

    def _update_temporal_tracks(self, candidates: List[dict]) -> List[dict]:
        """시계열 트랙 업데이트 및 n-프레임 이상 연속 검증된 객체 반환"""
        matched_ids: Set[int] = set()
        confirmed_candidates: List[dict] = []

        for candidate in candidates:
            track_id = self._match_candidate(candidate, matched_ids)

            if track_id is not None:
                track = self.temporal_tracks[track_id]
                track['bbox'] = candidate['bbox']
                track['area'] = candidate['area']
                track['aspect_ratio'] = candidate['aspect_ratio']
                track['hit_count'] += 1
                track['missed_count'] = 0

                matched_ids.add(track_id)

                if track['hit_count'] >= self.confirm_frames:
                    confirmed = dict(candidate)
                    confirmed['candidate_id'] = track_id
                    confirmed['temporal_hits'] = track['hit_count']
                    confirmed_candidates.append(confirmed)

            else:
                track_id = self.next_track_id
                self.next_track_id += 1

                self.temporal_tracks[track_id] = {
                    'bbox': candidate['bbox'],
                    'area': candidate['area'],
                    'aspect_ratio': candidate['aspect_ratio'],
                    'hit_count': 1,
                    'missed_count': 0
                }

                matched_ids.add(track_id)

        # 미매칭 트랙 정리 (missed_count 초과 시 삭제)
        delete_ids = []
        for track_id, track in self.temporal_tracks.items():
            if track_id in matched_ids:
                continue

            track['missed_count'] += 1
            if track['missed_count'] > self.max_missed_frames:
                delete_ids.append(track_id)

        for track_id in delete_ids:
            del self.temporal_tracks[track_id]

        return confirmed_candidates

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """RGB/Depth 수신 및 HSV 기반 화재 영역 추출 Callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB Image Conversion Failed: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 화재 색상 범위 (HSV 영역 1 & 영역 2)
        lower_bound1 = np.array([0, 70, 70], dtype=np.uint8)
        upper_bound1 = np.array([10, 255, 255], dtype=np.uint8)

        lower_bound2 = np.array([170, 120, 120], dtype=np.uint8)
        upper_bound2 = np.array([179, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 형태학적 연산 (Morphology Open / Close)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[dict] = []
        min_area = 500.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = float(h / float(w))
            if aspect_ratio < 0.25:
                continue

            candidates.append({
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'area': round(area, 1),
                'aspect_ratio': round(aspect_ratio, 3)
            })

        confirmed_candidates = self._update_temporal_tracks(candidates)

        stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        # ── /fire_candidates 토픽 발행 ─────────────────────────────────
        candidate_msg = String()
        candidate_msg.data = json.dumps({
            'header': {
                'stamp': stamp,
                'frame_id': rgb_msg.header.frame_id
            },
            'candidates': confirmed_candidates
        }, ensure_ascii=False)

        self.candidate_pub.publish(candidate_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FireDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Fire Detection Node stopped.')
    except Exception as e:
        node.get_logger().error(f'Exception in Fire Detection Node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()