#!/usr/bin/env python3
"""
입력:
    /camera/image_raw

출력:
    /camera/fall_detection/image
    /fall_alarm
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class FallDetectionNode(Node):

    def __init__(self):
        super().__init__('fall_detection_node')

        self.bridge = CvBridge()

        # ========================================================
        # YOLO Pose Model
        # ========================================================

        self.get_logger().info('Loading YOLOv8-pose model...')
        self.model = YOLO('/workspace/models/fall_yolov8n_pose/best.pt')
        self.get_logger().info('YOLOv8-pose model loaded successfully.')

        # ========================================================
        # Thresholds
        # ========================================================

        self.CONF_THRESHOLD = 0.5
        self.KEYPOINT_CONF_THRESHOLD = 0.5
        self.HORIZONTAL_ANGLE_THRESHOLD = 35.0

        # ========================================================
        # ROS Subscribers / Publishers
        # ========================================================

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/camera/fall_detection/image', 10)
        self.alert_pub = self.create_publisher(Bool, '/fall_alarm', 10)

        # ========================================================
        # Tracking
        # ========================================================

        self.tracking_history = {}
        self.TRACKING_WARMUP_FRAMES = 10

        # Detection loss 허용 프레임
        self.MAX_MISSED_FRAMES = 5
        self.missed_frames = {}

        # ========================================================
        # Initial State Voting
        # ========================================================

        self.initial_fall_votes = {}
        self.INITIAL_FALL_RATIO = 0.5

        # ========================================================
        # Fall Detection
        # ========================================================

        self.fall_history = {}
        self.FALL_THRESHOLD_FRAMES = 20

        # ========================================================
        # Track State
        # ========================================================

        self.track_states = {}

        self.get_logger().info(f'Confidence threshold: {self.CONF_THRESHOLD}')
        self.get_logger().info(f'Keypoint confidence threshold: {self.KEYPOINT_CONF_THRESHOLD}')
        self.get_logger().info(f'Horizontal angle threshold: {self.HORIZONTAL_ANGLE_THRESHOLD} deg')
        self.get_logger().info(f'Max missed frames: {self.MAX_MISSED_FRAMES}')
        self.get_logger().info('Fall Detection Node has been started.')

    # ============================================================
    # Keypoint Center
    # ============================================================

    def get_center(self, kpt, ids):
        """여러 keypoint의 평균 좌표를 계산."""

        points = []
        num_kpts = len(kpt)

        for idx in ids:
            if idx >= num_kpts:
                continue

            x, y, conf = kpt[idx]

            if conf >= self.KEYPOINT_CONF_THRESHOLD:
                points.append((float(x), float(y)))

        if not points:
            return None

        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)

        return x, y

    # ============================================================
    # Keypoint Geometry
    # ============================================================

    def get_geometry_points(self, kpt):
        """
        12 / 17 keypoint 구조에 맞춰
        Shoulder Center
        Hip Center
        Knee Center
        추출.
        """

        num_kpts = len(kpt)

        # Custom 12 Keypoint
        if num_kpts == 12:
            shoulder_ids = [1, 2]
            hip_ids = [7]
            knee_ids = [8, 9]

        # COCO 17 Keypoint
        elif num_kpts >= 17:
            shoulder_ids = [5, 6]
            hip_ids = [11, 12]
            knee_ids = [13, 14]

        else:
            return None

        shoulder = self.get_center(kpt, shoulder_ids)
        hip = self.get_center(kpt, hip_ids)
        knee = self.get_center(kpt, knee_ids)

        if shoulder is None or hip is None or knee is None:
            return None

        return shoulder, hip, knee

    # ============================================================
    # Line Angle
    # ============================================================

    def calculate_line_angle(self, p1, p2):
        """
        두 점을 연결하는 선의 수평 기준 각도.

        0°  = 수평
        90° = 수직
        """

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        angle = math.degrees(math.atan2(dy, dx))
        angle = abs(angle)

        if angle > 90.0:
            angle = 180.0 - angle

        return angle

    # ============================================================
    # Geometry Validation
    # ============================================================

    def calculate_geometry(self, kpt):
        """
        Shoulder -> Hip
        Hip -> Knee

        두 구간이 모두 수평에 가까운지 확인.
        """

        points = self.get_geometry_points(kpt)

        if points is None:
            return False, None, None, None

        shoulder, hip, knee = points

        shoulder_hip_angle = self.calculate_line_angle(shoulder, hip)
        hip_knee_angle = self.calculate_line_angle(hip, knee)

        if shoulder_hip_angle is None or hip_knee_angle is None:
            return False, shoulder_hip_angle, hip_knee_angle, points

        geometry_fall = (
            shoulder_hip_angle <= self.HORIZONTAL_ANGLE_THRESHOLD
            and hip_knee_angle <= self.HORIZONTAL_ANGLE_THRESHOLD
        )

        return geometry_fall, shoulder_hip_angle, hip_knee_angle, points

    # ============================================================
    # Geometry Visualization
    # ============================================================

    def draw_geometry(self, image, points, color):

        if points is None:
            return

        shoulder, hip, knee = points

        s = (int(shoulder[0]), int(shoulder[1]))
        h = (int(hip[0]), int(hip[1]))
        k = (int(knee[0]), int(knee[1]))

        cv2.line(image, s, h, color, 3)
        cv2.line(image, h, k, color, 3)

        cv2.circle(image, s, 6, color, -1)
        cv2.circle(image, h, 6, color, -1)
        cv2.circle(image, k, 6, color, -1)

    # ============================================================
    # Image Callback
    # ============================================================

    def image_callback(self, msg):

        # ========================================================
        # 1. ROS Image -> OpenCV
        # ========================================================

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # ========================================================
        # 2. YOLO Tracking
        # ========================================================

        results = self.model.track(
            cv_image,
            persist=True,
            conf=self.CONF_THRESHOLD,
            verbose=False
        )

        current_track_ids = set()
        fall_detected_global = False

        if results and len(results) > 0:

            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is not None and keypoints is not None:

                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                kpts = keypoints.data.cpu().numpy()

                # ====================================================
                # 3. Person Loop
                # ====================================================

                for i in range(len(xyxy)):

                    x1, y1, x2, y2 = map(int, xyxy[i])

                    track_id = track_ids[i] if i < len(track_ids) else -1

                    if track_id == -1:
                        continue

                    confidence = float(confidences[i]) if i < len(confidences) else 0.0

                    if confidence < self.CONF_THRESHOLD:
                        continue

                    # 현재 프레임에서 검출됨
                    current_track_ids.add(track_id)

                    # Detection loss 카운터 초기화
                    self.missed_frames[track_id] = 0

                    kpt = kpts[i]
                    cls_id = classes[i] if i < len(classes) else 0

                    # ====================================================
                    # 4. YOLO Fall
                    # ====================================================

                    yolo_falling = cls_id == 0

                    # ====================================================
                    # 5. Keypoint Geometry
                    # ====================================================

                    (
                        geometry_fall,
                        shoulder_hip_angle,
                        hip_knee_angle,
                        geometry_points
                    ) = self.calculate_geometry(kpt)

                    is_falling = yolo_falling and geometry_fall

                    # ====================================================
                    # 6. Tracking Frame Count
                    # ====================================================

                    self.tracking_history[track_id] = self.tracking_history.get(track_id, 0) + 1
                    tracking_frames = self.tracking_history[track_id]

                    # ====================================================
                    # 7. Track Initialization
                    # ====================================================

                    if track_id not in self.initial_fall_votes:
                        self.initial_fall_votes[track_id] = 0

                    if track_id not in self.fall_history:
                        self.fall_history[track_id] = 0

                    if track_id not in self.track_states:
                        self.track_states[track_id] = "tracking"

                    # ====================================================
                    # 8. Warm-up
                    # ====================================================

                    if tracking_frames <= self.TRACKING_WARMUP_FRAMES:

                        if is_falling:
                            self.initial_fall_votes[track_id] += 1

                        color = (255, 255, 0)

                        label = (
                            f"ID:{track_id} "
                            f"TRACKING "
                            f"({tracking_frames}/{self.TRACKING_WARMUP_FRAMES}) "
                            f"Conf:{confidence:.2f}"
                        )

                    # ====================================================
                    # 9. Initial State
                    # ====================================================

                    elif self.track_states[track_id] == "tracking":

                        fall_votes = self.initial_fall_votes[track_id]
                        fall_ratio = fall_votes / float(self.TRACKING_WARMUP_FRAMES)

                        if fall_ratio >= self.INITIAL_FALL_RATIO:

                            self.track_states[track_id] = "unsafe"
                            self.fall_history[track_id] = self.FALL_THRESHOLD_FRAMES
                            fall_detected_global = True
                            color = (0, 0, 255)

                            label = (
                                f"ID:{track_id} "
                                f"UNSAFE (Fall) "
                                f"Conf:{confidence:.2f}"
                            )

                        else:

                            self.track_states[track_id] = "safe"
                            self.fall_history[track_id] = 0
                            color = (0, 255, 0)

                            label = (
                                f"ID:{track_id} "
                                f"SAFE (Normal) "
                                f"Conf:{confidence:.2f}"
                            )

                    # ====================================================
                    # 10. Fall Detection
                    # ====================================================

                    else:

                        current_state = self.track_states[track_id]

                        # ----------------------------------------------
                        # UNSAFE
                        # ----------------------------------------------

                        if current_state == "unsafe":

                            if is_falling:
                                self.fall_history[track_id] = min(
                                    self.FALL_THRESHOLD_FRAMES,
                                    self.fall_history[track_id] + 1
                                )
                            else:
                                self.fall_history[track_id] = max(
                                    0,
                                    self.fall_history[track_id] - 2
                                )

                            if self.fall_history[track_id] == 0:

                                self.track_states[track_id] = "safe"
                                color = (0, 255, 0)

                                label = (
                                    f"ID:{track_id} "
                                    f"SAFE (Normal) "
                                    f"Conf:{confidence:.2f}"
                                )

                            else:

                                fall_detected_global = True
                                color = (0, 0, 255)

                                label = (
                                    f"ID:{track_id} "
                                    f"UNSAFE (Fall) "
                                    f"Conf:{confidence:.2f}"
                                )

                        # ----------------------------------------------
                        # SAFE
                        # ----------------------------------------------

                        else:

                            if is_falling:
                                self.fall_history[track_id] += 1
                            else:
                                self.fall_history[track_id] = max(
                                    0,
                                    self.fall_history[track_id] - 2
                                )

                            if self.fall_history[track_id] >= self.FALL_THRESHOLD_FRAMES:

                                self.track_states[track_id] = "unsafe"
                                fall_detected_global = True
                                color = (0, 0, 255)

                                label = (
                                    f"ID:{track_id} "
                                    f"UNSAFE (Fall) "
                                    f"Conf:{confidence:.2f}"
                                )

                            else:

                                color = (0, 255, 0)

                                label = (
                                    f"ID:{track_id} "
                                    f"SAFE (Normal) "
                                    f"Conf:{confidence:.2f}"
                                )

                    # ====================================================
                    # 11. Bounding Box
                    # ====================================================

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        cv_image,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # ====================================================
                    # 12. Geometry Information
                    # ====================================================

                    if shoulder_hip_angle is not None and hip_knee_angle is not None:

                        geometry_text = (
                            f"S-H:{shoulder_hip_angle:.1f} "
                            f"H-K:{hip_knee_angle:.1f}"
                        )

                        cv2.putText(
                            cv_image,
                            geometry_text,
                            (x1, min(cv_image.shape[0] - 10, y2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                    # ====================================================
                    # 13. Skeleton
                    # ====================================================

                    num_kpts = len(kpt)

                    if num_kpts == 12:

                        skeleton_edges = [
                            (1, 3),
                            (3, 5),
                            (2, 4),
                            (4, 6),
                            (1, 2),
                            (1, 7),
                            (2, 7),
                            (7, 8),
                            (8, 10),
                            (7, 9),
                            (9, 11)
                        ]

                    else:

                        skeleton_edges = [
                            (5, 7),
                            (7, 9),
                            (6, 8),
                            (8, 10),
                            (5, 6),
                            (5, 11),
                            (6, 12),
                            (11, 12),
                            (11, 13),
                            (13, 15),
                            (12, 14),
                            (14, 16)
                        ]

                    # ====================================================
                    # 14. Keypoint 표시
                    # ====================================================

                    for pt_idx in range(num_kpts):

                        px, py, conf = kpt[pt_idx]

                        if conf > self.KEYPOINT_CONF_THRESHOLD:
                            cv2.circle(
                                cv_image,
                                (int(px), int(py)),
                                4,
                                color,
                                -1
                            )

                    # ====================================================
                    # 15. Geometry 표시
                    # ====================================================

                    self.draw_geometry(cv_image, geometry_points, color)

                    # ====================================================
                    # 16. Skeleton 연결
                    # ====================================================

                    for edge in skeleton_edges:

                        p1, p2 = edge

                        if p1 < num_kpts and p2 < num_kpts:

                            x1_k, y1_k, conf1 = kpt[p1]
                            x2_k, y2_k, conf2 = kpt[p2]

                            if (
                                conf1 > self.KEYPOINT_CONF_THRESHOLD
                                and conf2 > self.KEYPOINT_CONF_THRESHOLD
                            ):
                                cv2.line(
                                    cv_image,
                                    (int(x1_k), int(y1_k)),
                                    (int(x2_k), int(y2_k)),
                                    color,
                                    2
                                )

        # ========================================================
        # 17. Detection Loss / Track 유지
        # ========================================================

        all_track_ids = set(self.tracking_history.keys())

        for track_id in all_track_ids:

            # 이번 프레임에 검출됨
            if track_id in current_track_ids:
                self.missed_frames[track_id] = 0
                continue

            # Detection loss
            self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1

            # 일정 프레임까지 기존 track 유지
            if self.missed_frames[track_id] <= self.MAX_MISSED_FRAMES:
                continue

            # 오래 사라지면 track 삭제
            self.tracking_history.pop(track_id, None)
            self.initial_fall_votes.pop(track_id, None)
            self.fall_history.pop(track_id, None)
            self.track_states.pop(track_id, None)
            self.missed_frames.pop(track_id, None)

        # ========================================================
        # 18. 화면 출력
        # ========================================================

        cv2.imshow("Fall Detection Monitor", cv_image)
        cv2.waitKey(1)

        # ========================================================
        # 19. 결과 이미지 Publish
        # ========================================================

        try:

            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)

        except Exception as e:

            self.get_logger().error(f"Failed to publish image: {e}")

        # ========================================================
        # 20. Fall Alarm
        # ========================================================

        alert_msg = Bool()
        alert_msg.data = fall_detected_global
        self.alert_pub.publish(alert_msg)


def main(args=None):

    rclpy.init(args=args)
    node = FallDetectionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Fall Detection Node stopped cleanly')

    except Exception as e:
        node.get_logger().error(f'Exception in Fall Detection Node: {e}')

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()