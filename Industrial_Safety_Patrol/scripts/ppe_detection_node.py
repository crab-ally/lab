#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

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

        # 기본 YOLOv8 small 모델 로드
        #
        # 현재 yolov8s.pt는 COCO 데이터셋으로 학습된 범용 모델
        # -> person 탐지는 가능
        # -> helmet / vest는 탐지 불가능
        #
        # 추후 형태로 PPE 전용 학습 모델로 교체 필요
        self.model = YOLO('/workspace/models/ppe_yolov8n/weights/best.pt')
        self.get_logger().info('YOLOv8 model loaded successfully.')

        ####################################################
        # YOLO 클래스 ID 설정
        ####################################################

        self.CLASS_PERSON = 0
        self.CLASS_HELMET = 1
        self.CLASS_VEST = 2

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

        ####################################################
        # YOLO 결과 처리
        ####################################################

        for result in results:

            boxes = result.boxes

            # 객체 종류별 Bounding Box 저장
            persons = []
            helmets = []
            vests = []

            ################################################
            # 탐지된 객체 분류
            ################################################

            for box in boxes:

                cls_id = int(box.cls[0]) # 클래스 번호
                conf = float(box.conf[0]) # confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding Box 좌표

                # 신뢰도가 낮은 탐지는 제외
                if conf < 0.5:
                    continue

                bbox = (x1, y1, x2, y2)

                # 사람 객체
                if cls_id == self.CLASS_PERSON:
                    persons.append(bbox)

                # 안전모 객체
                elif cls_id == self.CLASS_HELMET:
                    helmets.append(bbox)

                # 안전조끼 객체
                elif cls_id == self.CLASS_VEST:
                    vests.append(bbox)

            ################################################
            # 사람별 PPE 착용 여부 판단
            ################################################

            for (px1, py1, px2, py2) in persons:
                has_helmet = False
                has_vest = False

                # 사람 영역과 안전모 Bounding Box 비교
                for (hx1, hy1, hx2, hy2) in helmets:

                    # 두 박스가 겹치는지 확인
                    if not (
                        hx2 < px1 or
                        hx1 > px2 or
                        hy2 < py1 or
                        hy1 > py2
                    ):
                        has_helmet = True
                        break

                # 사람 영역과 조끼 Bounding Box 비교
                for (vx1, vy1, vx2, vy2) in vests:

                    if not (
                        vx2 < px1 or
                        vx1 > px2 or
                        vy2 < py1 or
                        vy1 > py2
                    ):
                        has_vest = True
                        break

                ################################################
                # PPE 안전 판정
                ################################################

                is_safe = has_helmet and has_vest

                if not is_safe:

                    unsafe_detected = True

                    # 위험 상태: 빨간색
                    color = (0, 0, 255)
                    label = "UNSAFE (Missing PPE)"

                else:

                    # 정상 상태: 초록색
                    color = (0, 255, 0)
                    label = "SAFE (PPE OK)"

                ################################################
                # 사람 Bounding Box 표시
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