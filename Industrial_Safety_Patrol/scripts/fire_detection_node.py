#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')
        
        # 카메라 원본 이미지 구독
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        # 화재 영역 표시 결과 이미지 Publish
        self.image_pub = self.create_publisher(
            Image,
            '/camera/fire_detection/image',
            10
        )
        # 화재 발생 여부 Publish (T/F)
        self.alert_pub = self.create_publisher(
            Bool,
            '/fire_alarm',
            10
        )
        
        self.bridge = CvBridge()
        
        self.get_logger().info('Fire Detection Node has been started.')

    def image_callback(self, msg):
        # ==========================================================
        # 1. ROS Image 메시지 → OpenCV 이미지 변환
        # ==========================================================
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # ==========================================================
        # 2. BGR → HSV 색 공간 변환
        # ==========================================================
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # ==========================================================
        # 3. 화재 색상 영역 추출
        # ==========================================================
        
        # Range 1: Red ~ Yellow/Orange
        lower_bound1 = np.array([0, 120, 120])
        upper_bound1 = np.array([30, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        
        # Range 2: Red (170-179)
        lower_bound2 = np.array([170, 120, 120])
        upper_bound2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)

        # ==========================================================
        # 4. Noise 제거 (Morphological Filtering)
        # ==========================================================
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # ==========================================================
        # 5. Mask에서 화재 영역 Contour 찾기
        # ==========================================================
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False

        # ==========================================================
        # 6. Contour 크기 검사 후 화재 판단
        # ==========================================================
        min_area = 500  # Threshold for fire area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(cv_image, 'FIRE DETECTED', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                fire_detected = True

        # ==========================================================
        # 7. 결과 영상 Publish
        # ==========================================================
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)

            # 모니터링용 OpenCV 디스플레이 창 띄우기
            cv2.imshow("Fire Detection Monitor", cv_image)
            cv2.waitKey(1)  # 영상 갱신을 위해 필수

            if fire_detected:
                self.get_logger().info('fire detected!')
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

        # ==========================================================
        # 8. 화재 Alarm Publish
        # ========================================================== 
        alert_msg = Bool()
        alert_msg.data = fire_detected
        self.alert_pub.publish(alert_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Fire Detection Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Exception in Fire Detection Node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
