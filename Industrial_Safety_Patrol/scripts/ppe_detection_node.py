#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from geometry_msgs.msg import Twist, PoseStamped

class PPEDetectionNode(Node):
    """
    PPE(Personal Protective Equipment) 착용 여부 탐지 ROS2 Node

    입력:
        /camera/image_raw - 로봇 카메라에서 전달되는 원본 이미지

    출력:
        /camera/ppe_detection/image - PPE 탐지 결과가 표시된 이미지
        /ppe_alarm - PPE 미착용 감지 여부 (T/F)
    """

    def __init__(self):
        super().__init__('ppe_detection_node')

        self.bridge = CvBridge()
        self.window_open = False

        ####################################################
        # YOLO 모델 초기화
        ####################################################

        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('/workspace/models/ppe_forklift_yolov8n/best.pt')
        self.get_logger().info('YOLOv8 model loaded successfully.')

        ####################################################
        # YOLO 클래스 ID 설정
        ####################################################

        self.CLASS_PERSON = 0
        self.CLASS_HELMET = 1
        self.CLASS_VEST = 2
        self.CLASS_FORKLIFT = 3

        ####################################################
        # DeepSORT 초기화
        ####################################################

        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

        ####################################################
        # ROS Subscriber
        ####################################################

        # 카메라 원본 영상 입력
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        ####################################################
        # ROS Publisher
        ####################################################

        # PPE 탐지 결과 영상 출력
        self.image_pub = self.create_publisher(
            Image,
            '/camera/ppe_detection/image',
            10
        )

        # PPE 미착용 Alarm 출력 (T/F)
        self.alert_pub = self.create_publisher(
            Bool,
            '/ppe_alarm',
            10
        )

        self.get_logger().info('PPE Detection Node has been started.')

        ####################################################
        # Forklift Control Initialization
        ####################################################
        self.fl1_pose = None
        self.fl2_pose = None
        
        # 임의의 왕복 경로
        self.fl1_waypoints = [
            [-8.0, 8.0],
            [-3.0, 8.0],
            [-3.0, 3.0],
            [-8.0, 3.0],
        ]
        self.fl2_waypoints = [
            [3.0, 8.0],
            [3.0, 3.0],
            [8.0, 3.0],
            [8.0, 8.0],
        ]
        
        self.fl1_wp_idx = 0
        self.fl2_wp_idx = 0
        self.fl_speed = 1.0
        
        # Subscribers for Forklift Pose
        self.fl1_pose_sub = self.create_subscription(PoseStamped, '/forklift_1/pose', self.fl1_pose_callback, 10)
        self.fl2_pose_sub = self.create_subscription(PoseStamped, '/forklift_2/pose', self.fl2_pose_callback, 10)
        
        # Publishers for Forklift Cmd Vel
        self.fl1_cmd_pub = self.create_publisher(Twist, '/forklift_1/cmd_vel', 10)
        self.fl2_cmd_pub = self.create_publisher(Twist, '/forklift_2/cmd_vel', 10)
        
        # Timer for Forklift Control (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_timer_callback)

    def fl1_pose_callback(self, msg: PoseStamped):
        self.fl1_pose = np.array([msg.pose.position.x, msg.pose.position.y])
        
    def fl2_pose_callback(self, msg: PoseStamped):
        self.fl2_pose = np.array([msg.pose.position.x, msg.pose.position.y])

    def _compute_forklift_twist(self, current_pose, waypoints, wp_idx):
        twist = Twist()
        if current_pose is None:
            return twist, wp_idx
            
        target = np.array(waypoints[wp_idx])
        diff = target - current_pose
        dist = np.linalg.norm(diff)
        
        if dist < 0.5:
            wp_idx = (wp_idx + 1) % len(waypoints)
            target = np.array(waypoints[wp_idx])
            diff = target - current_pose
            dist = np.linalg.norm(diff)
            
        if dist >= 0.5:
            dir_vec = diff / dist # 단위벡터 변환
            twist.linear.x = float(dir_vec[0] * self.fl_speed)
            twist.linear.y = float(dir_vec[1] * self.fl_speed)
            
        return twist, wp_idx

    def control_timer_callback(self):
        # Forklift 1
        fl1_twist, self.fl1_wp_idx = self._compute_forklift_twist(self.fl1_pose, self.fl1_waypoints, self.fl1_wp_idx)
        self.fl1_cmd_pub.publish(fl1_twist)
        
        # Forklift 2
        fl2_twist, self.fl2_wp_idx = self._compute_forklift_twist(self.fl2_pose, self.fl2_waypoints, self.fl2_wp_idx)
        self.fl2_cmd_pub.publish(fl2_twist)


    def image_callback(self, msg):
        """
        카메라 이미지 Callback

        동작 순서:
        1. ROS Image → OpenCV 변환
        2. YOLO 객체 탐지
        3. 사람별 PPE 착용 여부 판단
        4. 결과 영상 Publish
        5. Alarm Publish
        """

        ####################################################
        # ROS Image 메시지를 OpenCV 이미지로 변환
        ####################################################

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        ####################################################
        # YOLO 객체 탐지 수행
        ####################################################

        results = self.model(cv_image, verbose=False)
        unsafe_detected = False

        for result in results:
            self.get_logger().info(f"Detected boxes: {len(result.boxes)}")

        ####################################################
        # YOLO 결과 처리
        ####################################################

        bbs = []
        helmets = []
        vests = []

        for result in results:

            boxes = result.boxes

            ################################################
            # 탐지된 객체 분류
            ################################################

            for box in boxes:

                cls_id = int(box.cls[0]) # 클래스 번호
                conf = float(box.conf[0]) # confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding Box 좌표

                # 디버깅용 추가
                # self.get_logger().info(
                #     f"class={cls_id}, name={self.model.names[cls_id]}, conf={conf:.2f}"
                # )

                bbox = (x1, y1, x2, y2)

                # 사람 또는 지게차 객체 (Tracking 대상)
                if cls_id == self.CLASS_PERSON or cls_id == self.CLASS_FORKLIFT:
                    w = x2 - x1
                    h = y2 - y1
                    bbs.append(([x1, y1, w, h], conf, cls_id))

                # 안전모 객체
                elif cls_id == self.CLASS_HELMET:
                    helmets.append(bbox)

                # 안전조끼 객체
                elif cls_id == self.CLASS_VEST:
                    vests.append(bbox)

        ################################################
        # DeepSORT Tracking 수행
        ################################################
        
        tracks = self.tracker.update_tracks(bbs, frame=cv_image)

        ################################################
        # Tracking 결과 시각화 및 PPE 검증
        ################################################

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            px1, py1, px2, py2 = map(int, ltrb)
            cls_id = track.get_det_class()
            
            # 클래스가 string으로 리턴되는 경우 방어 코드
            if isinstance(cls_id, str):
                cls_id = int(cls_id)
                
            if cls_id == self.CLASS_PERSON:
                has_helmet = False
                has_vest = False

                # 헬멧 중심점이 사람 박스 내부에 있는지 확인
                for (hx1, hy1, hx2, hy2) in helmets:
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0
                    if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                        has_helmet = True
                        break

                # 조끼 중심점이 사람 박스 내부에 있는지 확인
                for (vx1, vy1, vx2, vy2) in vests:
                    vcx = (vx1 + vx2) / 2.0
                    vcy = (vy1 + vy2) / 2.0
                    if px1 <= vcx <= px2 and py1 <= vcy <= py2:
                        has_vest = True
                        break

                ################################################
                # PPE 안전 판정
                ################################################

                is_safe = has_helmet and has_vest

                if not is_safe:
                    unsafe_detected = True
                    color = (0, 0, 255)
                    label = f"Worker ID {track_id} (UNSAFE)"
                else:
                    color = (0, 255, 0)
                    label = f"Worker ID {track_id} (SAFE)"
                    
            elif cls_id == self.CLASS_FORKLIFT:
                color = (0, 165, 255) # 주황색 (BGR)
                label = f"Forklift ID {track_id}"
                
            else:
                continue

            ################################################
            # 사람/지게차 추적 Bounding Box 표시
            ################################################

            cv2.rectangle(
                cv_image,
                (px1, py1),
                (px2, py2),
                color,
                2
            )

            cv2.putText(
                cv_image,
                label,
                (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        ################################################
        # Helmet / Vest 탐지 결과 표시
        ################################################

        for (hx1, hy1, hx2, hy2) in helmets:

            cv2.rectangle(
                cv_image,
                (hx1, hy1),
                (hx2, hy2),
                (255,0,0),
                2
            )

            cv2.putText(
                cv_image,
                "Helmet",
                (hx1, hy1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,0,0),
                1
            )

        for (vx1, vy1, vx2, vy2) in vests:

            cv2.rectangle(
                cv_image,
                (vx1, vy1),
                (vx2, vy2),
                (255,255,0),
                2
            )

            cv2.putText(
                cv_image,
                "Vest",
                (vx1, vy1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,0),
                1
            )

        ####################################################
        # 실시간 모니터링 화면 출력
        ####################################################

        cv2.imshow("PPE Detection Monitor", cv_image)
        cv2.waitKey(1)

        ####################################################
        # 결과 이미지 ROS Publish
        ####################################################

        try:
            processed_msg = self.bridge.cv2_to_imgmsg(
                cv_image,
                encoding='bgr8'
            )
            processed_msg.header = msg.header # 원본 이미지 timestamp 유지
            self.image_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(
                f"Failed to publish image: {e}"
            )

        ####################################################
        # PPE Alarm Publish
        ####################################################
        alert_msg = Bool()
        alert_msg.data = unsafe_detected
        self.alert_pub.publish(alert_msg)


def main(args=None):
    rclpy.init(args=args) # ROS2 초기화
    node = PPEDetectionNode() # Node 생성

    try:
        rclpy.spin(node) # Callback 실행
    except KeyboardInterrupt:
        node.get_logger().info('PPE Detection Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Exception in PPE Detection Node: {e}')
    finally:
        cv2.destroyAllWindows() # OpenCV 창 종료
        node.destroy_node() # Node 제거
        rclpy.shutdown()

if __name__ == '__main__':

    main()