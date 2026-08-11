#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class FallDetectionNode(Node):
    """
    YOLO Pose 기반 쓰러짐 감지(Fall Detection) ROS2 Node

    작업자의 자세 클래스:
        0: fallen
        1: standing
        2: bending
        3: sitting

    동작:
        1. 새로운 사람이 Tracking 시작
        2. 일정 프레임 동안 상태 관찰
        3. Warm-up 종료 후 초기 상태 확정
           - fallen이 대부분 → UNSAFE
           - 정상 자세가 대부분 → SAFE
        4. 이후 정상 상태에서 실제로 넘어지는 경우
           Fall History를 이용하여 UNSAFE 판정

    입력:
        /camera/image_raw

    출력:
        /camera/fall_detection/image
        /fall_alarm
    """

    def __init__(self):
        super().__init__('fall_detection_node')

        self.bridge = CvBridge()

        # ========================================================
        # YOLO Pose Model
        # ========================================================

        self.get_logger().info(
            'Loading YOLOv8-pose model...'
        )

        self.model = YOLO(
            '/workspace/models/fall_yolov8n_pose/best.pt'
        )

        self.get_logger().info(
            'YOLOv8-pose model loaded successfully.'
        )

        # ========================================================
        # ROS Subscribers / Publishers
        # ========================================================

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/camera/fall_detection/image',
            10
        )

        self.alert_pub = self.create_publisher(
            Bool,
            '/fall_alarm',
            10
        )

        # ========================================================
        # Tracking
        # ========================================================

        # {track_id: tracking_frames}
        self.tracking_history = {}

        # Tracking 시작 후 상태를 관찰하는 프레임 수
        self.TRACKING_WARMUP_FRAMES = 10

        # ========================================================
        # Initial State Voting
        # ========================================================

        # {track_id: fallen_count}
        #
        # Warm-up 10프레임 동안 fallen이 몇 번 검출됐는지 저장
        self.initial_fall_votes = {}

        # Warm-up 프레임 중 이 비율 이상 fallen이면
        # 처음부터 쓰러져 있었다고 판단
        self.INITIAL_FALL_RATIO = 0.5

        # ========================================================
        # Fall Detection
        # ========================================================

        # {track_id: fall_frames}
        self.fall_history = {}

        # 정상 상태 → 낙상 상태로 전환할 때 필요한 누적 프레임
        self.FALL_THRESHOLD_FRAMES = 20

        # ========================================================
        # 현재 사람의 상태
        # ========================================================

        # {track_id: "tracking" / "safe" / "unsafe"}
        self.track_states = {}

        self.get_logger().info(
            'Fall Detection Node has been started.'
        )

    def image_callback(self, msg):

        # ========================================================
        # 1. ROS Image → OpenCV
        # ========================================================

        try:

            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to convert image: {e}"
            )

            return

        # ========================================================
        # 2. YOLO Tracking
        # ========================================================

        results = self.model.track(
            cv_image,
            persist=True,
            verbose=False
        )

        current_track_ids = set()

        # 이번 프레임 전체 Fall Alarm
        fall_detected_global = False

        if results and len(results) > 0:

            result = results[0]

            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is not None and keypoints is not None:

                # ------------------------------------------------
                # Track ID
                # ------------------------------------------------

                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else []
                )

                # ------------------------------------------------
                # Class ID
                # ------------------------------------------------

                classes = (
                    boxes.cls.int().cpu().tolist()
                    if boxes.cls is not None
                    else []
                )

                # ------------------------------------------------
                # Bounding Box
                # ------------------------------------------------

                xyxy = boxes.xyxy.cpu().numpy()

                # ------------------------------------------------
                # Keypoints
                # ------------------------------------------------

                kpts = keypoints.data.cpu().numpy()

                # ====================================================
                # 3. 각 사람 처리
                # ====================================================

                for i in range(len(xyxy)):

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy[i]
                    )

                    track_id = (
                        track_ids[i]
                        if i < len(track_ids)
                        else -1
                    )

                    if track_id == -1:
                        continue

                    current_track_ids.add(track_id)

                    kpt = kpts[i]

                    cls_id = (
                        classes[i]
                        if i < len(classes)
                        else 0
                    )

                    # ====================================================
                    # 4. 현재 클래스
                    # ====================================================

                    # Custom Pose Model
                    #
                    # 0: fallen
                    # 1: standing
                    # 2: bending
                    # 3: sitting

                    is_falling = (cls_id == 0)

                    # ====================================================
                    # 5. Tracking Frame Count
                    # ====================================================

                    self.tracking_history[track_id] = (
                        self.tracking_history.get(track_id, 0) + 1
                    )

                    tracking_frames = (
                        self.tracking_history[track_id]
                    )

                    # ====================================================
                    # 6. 새로운 Track 초기화
                    # ====================================================

                    if track_id not in self.initial_fall_votes:

                        self.initial_fall_votes[track_id] = 0

                    if track_id not in self.fall_history:

                        self.fall_history[track_id] = 0

                    if track_id not in self.track_states:

                        self.track_states[track_id] = "tracking"

                    # ====================================================
                    # 7. Tracking Warm-up
                    # ====================================================

                    if tracking_frames <= self.TRACKING_WARMUP_FRAMES:

                        # --------------------------------------------
                        # fallen 검출 횟수 기록
                        # --------------------------------------------

                        if is_falling:

                            self.initial_fall_votes[track_id] += 1

                        # --------------------------------------------
                        # 아직 최종 판단하지 않음
                        # --------------------------------------------

                        color = (255, 255, 0)

                        label = (
                            f"ID:{track_id} "
                            f"TRACKING "
                            f"({tracking_frames}/"
                            f"{self.TRACKING_WARMUP_FRAMES})"
                        )

                    # ====================================================
                    # 8. Warm-up 종료 → 초기 상태 확정
                    # ====================================================

                    elif (
                        self.track_states[track_id]
                        == "tracking"
                    ):

                        fall_votes = (
                            self.initial_fall_votes[track_id]
                        )

                        fall_ratio = (
                            fall_votes
                            / float(self.TRACKING_WARMUP_FRAMES)
                        )

                        # --------------------------------------------
                        # 처음부터 쓰러져 있었음
                        # --------------------------------------------

                        if fall_ratio >= self.INITIAL_FALL_RATIO:

                            self.track_states[track_id] = "unsafe"

                            self.fall_history[track_id] = (
                                self.FALL_THRESHOLD_FRAMES
                            )

                            fall_detected_global = True

                            color = (0, 0, 255)

                            label = (
                                f"ID:{track_id} "
                                f"UNSAFE (Fall)"
                            )

                        # --------------------------------------------
                        # 정상 상태로 시작
                        # --------------------------------------------

                        else:

                            self.track_states[track_id] = "safe"

                            self.fall_history[track_id] = 0

                            color = (0, 255, 0)

                            label = (
                                f"ID:{track_id} "
                                f"SAFE (Normal)"
                            )

                    # ====================================================
                    # 9. Warm-up 이후 Fall Detection
                    # ====================================================

                    else:

                        current_state = (
                            self.track_states[track_id]
                        )

                        # ==================================================
                        # 현재 UNSAFE 상태
                        # ==================================================

                        if current_state == "unsafe":

                            # --------------------------------------------
                            # 이미 낙상으로 판정된 사람
                            # --------------------------------------------

                            if is_falling:

                                # 계속 fallen이면 UNSAFE 유지
                                self.fall_history[track_id] = min(
                                    self.FALL_THRESHOLD_FRAMES,
                                    self.fall_history[track_id] + 1
                                )

                            else:

                                # 정상 자세가 나오면 서서히 감소
                                self.fall_history[track_id] = max(
                                    0,
                                    self.fall_history[track_id] - 2
                                )

                            # 충분히 회복되면 SAFE
                            if (
                                self.fall_history[track_id] == 0
                            ):

                                self.track_states[track_id] = "safe"

                                color = (0, 255, 0)

                                label = (
                                    f"ID:{track_id} "
                                    f"SAFE (Normal)"
                                )

                            else:

                                fall_detected_global = True

                                color = (0, 0, 255)

                                label = (
                                    f"ID:{track_id} "
                                    f"UNSAFE (Fall)"
                                )

                        # ==================================================
                        # 현재 SAFE 상태
                        # ==================================================

                        else:

                            if is_falling:

                                # ----------------------------------------
                                # 정상 → fallen 전환 감시
                                # ----------------------------------------

                                self.fall_history[track_id] = (
                                    self.fall_history.get(track_id, 0) + 1
                                )

                            else:

                                # 정상 자세
                                self.fall_history[track_id] = max(
                                    0,
                                    self.fall_history.get(track_id, 0) - 2
                                )

                            # --------------------------------------------
                            # Fall 확정
                            # --------------------------------------------

                            if (
                                self.fall_history[track_id]
                                >= self.FALL_THRESHOLD_FRAMES
                            ):

                                self.track_states[track_id] = "unsafe"

                                fall_detected_global = True

                                color = (0, 0, 255)

                                label = (
                                    f"ID:{track_id} "
                                    f"UNSAFE (Fall)"
                                )

                            else:

                                color = (0, 255, 0)

                                label = (
                                    f"ID:{track_id} "
                                    f"SAFE (Normal)"
                                )

                    # ====================================================
                    # 10. Bounding Box
                    # ====================================================

                    cv2.rectangle(
                        cv_image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

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
                    # 11. Skeleton
                    # ====================================================

                    num_kpts = len(kpt)

                    if num_kpts == 12:

                        skeleton_edges = [

                            # Left arm
                            (1, 3),
                            (3, 5),

                            # Right arm
                            (2, 4),
                            (4, 6),

                            # Torso
                            (1, 2),
                            (1, 7),
                            (2, 7),

                            # Left leg
                            (7, 8),
                            (8, 10),

                            # Right leg
                            (7, 9),
                            (9, 11)
                        ]

                    else:

                        skeleton_edges = [

                            # Left arm
                            (5, 7),
                            (7, 9),

                            # Right arm
                            (6, 8),
                            (8, 10),

                            # Torso
                            (5, 6),
                            (5, 11),
                            (6, 12),
                            (11, 12),

                            # Left leg
                            (11, 13),
                            (13, 15),

                            # Right leg
                            (12, 14),
                            (14, 16)
                        ]

                    # ====================================================
                    # 12. Keypoint 표시
                    # ====================================================

                    for pt_idx in range(num_kpts):

                        px, py, conf = kpt[pt_idx]

                        if conf > 0.5:

                            cv2.circle(
                                cv_image,
                                (int(px), int(py)),
                                4,
                                color,
                                -1
                            )

                    # ====================================================
                    # 13. Skeleton 연결
                    # ====================================================

                    for edge in skeleton_edges:

                        p1, p2 = edge

                        if (
                            p1 < num_kpts
                            and p2 < num_kpts
                        ):

                            x1_k, y1_k, conf1 = kpt[p1]
                            x2_k, y2_k, conf2 = kpt[p2]

                            if (
                                conf1 > 0.5
                                and conf2 > 0.5
                            ):

                                cv2.line(
                                    cv_image,
                                    (int(x1_k), int(y1_k)),
                                    (int(x2_k), int(y2_k)),
                                    color,
                                    2
                                )

        # ========================================================
        # 14. Tracking 종료 ID 정리
        # ========================================================

        finished_track_ids = (
            set(self.tracking_history.keys())
            - current_track_ids
        )

        for track_id in finished_track_ids:

            self.tracking_history.pop(
                track_id,
                None
            )

            self.initial_fall_votes.pop(
                track_id,
                None
            )

            self.fall_history.pop(
                track_id,
                None
            )

            self.track_states.pop(
                track_id,
                None
            )

        # ========================================================
        # 15. 화면 출력
        # ========================================================

        cv2.imshow(
            "Fall Detection Monitor",
            cv_image
        )

        cv2.waitKey(1)

        # ========================================================
        # 16. 결과 이미지 Publish
        # ========================================================

        try:

            processed_msg = self.bridge.cv2_to_imgmsg(
                cv_image,
                encoding='bgr8'
            )

            processed_msg.header = msg.header

            self.image_pub.publish(
                processed_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to publish image: {e}"
            )

        # ========================================================
        # 17. Fall Alarm Publish
        # ========================================================

        alert_msg = Bool()

        alert_msg.data = fall_detected_global

        self.alert_pub.publish(
            alert_msg
        )


def main(args=None):

    rclpy.init(args=args)

    node = FallDetectionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        node.get_logger().info(
            'Fall Detection Node stopped cleanly'
        )

    except Exception as e:

        node.get_logger().error(
            f'Exception in Fall Detection Node: {e}'
        )

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
