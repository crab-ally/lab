#!/usr/bin/env python3
"""
Node1: FireDetectionNode

SUB:
    /camera/image_raw
    /camera/depth/image_raw
    /fire_tracks_3d
PUB:
    /fire_candidates
    /camera/fire_detection/image
    /fire_alarm
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters


class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')
        self.bridge = CvBridge()
        self.window_open = False
        self.validated_fires = []

        # RGB + Depth synchronization
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.candidate_pub = self.create_publisher(String, '/fire_candidates', 10)
        self.image_pub = self.create_publisher(Image, '/camera/fire_detection/image', 10)
        self.alert_pub = self.create_publisher(Bool, '/fire_alarm', 10)

        # Fire Fusion result
        self.fire_tracks_sub = self.create_subscription(
            String,
            '/fire_tracks_3d',
            self.fire_tracks_callback,
            10
        )

        self.get_logger().info('Fire Detection Node started.')

    def fire_tracks_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            self.validated_fires = payload.get('fires', [])
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Fire track JSON error: {e}')

    def image_callback(self, rgb_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # BGR -> HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red mask
        lower_bound1 = np.array([0, 70, 70])
        upper_bound1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)

        lower_bound2 = np.array([170, 120, 120])
        upper_bound2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        min_area = 500

        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            aspect_ratio = h / float(w)

            # 너무 넓고 낮은 영역 제거
            if aspect_ratio < 0.25:
                continue

            # BBox 내부 색상 영역 비율
            bbox_area = w * h
            fill_ratio = area / float(bbox_area)
            fill_ratio = max(0.0, min(1.0, fill_ratio))

            # 면적 기반 score
            area_score = min(area / 5000.0, 1.0)

            confidence = (
                0.6 * fill_ratio +
                0.4 * area_score
            )

            candidates.append({
                'candidate_id': index,
                'bbox': [
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                ],
                'confidence': round(float(confidence), 3),
                'area': round(float(area), 1),
                'aspect_ratio': round(float(aspect_ratio), 3)
            })

        # Fire candidates publish
        candidate_msg = String()
        candidate_msg.data = json.dumps({
            'header': {
                'stamp': (
                    rgb_msg.header.stamp.sec +
                    rgb_msg.header.stamp.nanosec * 1e-9
                ),
                'frame_id': rgb_msg.header.frame_id
            },
            'candidates': candidates
        }, ensure_ascii=False)

        self.candidate_pub.publish(candidate_msg)

        # Fusion을 통과한 화재만 최종 감지
        fire_detected = len(self.validated_fires) > 0

        if fire_detected:
            for fire in self.validated_fires:
                fire_id = fire.get('fire_id')
                confidence = fire.get('confidence', 0.0)

                self.get_logger().warn(
                    f'FIRE DETECTED '
                    f'id={fire_id}, '
                    f'confidence={confidence:.2f}, '
                    f'position={fire.get("position")}'
                )

            cv2.putText(
                cv_image,
                'FIRE DETECTED',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            cv2.imshow(
                'Fire Detection Monitor',
                cv_image
            )
            cv2.waitKey(1)
            self.window_open = True

        elif self.window_open:
            cv2.destroyAllWindows()
            self.window_open = False

        # Processed image
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(
                cv_image,
                encoding='bgr8'
            )
            processed_msg.header = rgb_msg.header
            self.image_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(
                f'Failed to publish image: {e}'
            )

        # Fire alarm
        alert_msg = Bool()
        alert_msg.data = fire_detected
        self.alert_pub.publish(alert_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Fire Detection Node stopped cleanly.'
        )
    except Exception as e:
        node.get_logger().error(
            f'Exception: {e}'
        )
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()