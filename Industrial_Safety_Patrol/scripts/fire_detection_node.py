#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

# RGB와 Depth 토픽을 시간 동기화하기 위한 라이브러리
import message_filters

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')
        
        self.bridge = CvBridge()
        self.window_open = False  # 모니터링 창 열림 상태 플래그
        
        # ==========================================================
        # 1. RGB & Depth 토픽 구독 및 시간 동기화 (ApproximateTimeSync)
        # ==========================================================
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        # RGB와 Depth의 발행 시간이 약간 다를 수 있으므로 ApproximateTimeSynchronizer 사용
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # 결과 및 알람 Publisher
        self.image_pub = self.create_publisher(Image, '/camera/fire_detection/image', 10)
        self.alert_pub = self.create_publisher(Bool, '/fire_alarm', 10)
        
        self.get_logger().info('Fire Detection Node (with Depth Filtering) has been started.')

    def image_callback(self, rgb_msg, depth_msg):
        # ==========================================================
        # 1. ROS Image 메시지 → OpenCV 이미지 변환
        # ==========================================================
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Depth 이미지는 보통 32FC1 (meters) 또는 16UC1 (millimeters)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # ==========================================================
        # 2. BGR → HSV 색 공간 변환
        # ==========================================================
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # ==========================================================
        # 3. 화재/소화전 붉은색 영역 추출
        # ==========================================================
        lower_bound1 = np.array([0, 120, 120])
        upper_bound1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        
        lower_bound2 = np.array([170, 120, 120])
        upper_bound2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        
        mask = cv2.bitwise_or(mask1, mask2)

        # ==========================================================
        # 4. 노이즈 제거 (Morphology)
        # ==========================================================
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # ==========================================================
        # 5. Contour 영역 탐색 및 Depth 검증
        # ==========================================================
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        min_area = 500  # 최소 면적 임계값

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
        
                roi_depth = depth_image[y:y+h, x:x+w]
                valid_depths = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0)]
        
                if len(valid_depths) < 10:
                    continue

                depth_std = np.std(valid_depths)
                depth_mean = np.mean(valid_depths) # 평균 거리
        
                if depth_image.dtype == np.uint16:
                    depth_std /= 1000.0
                    depth_mean /= 1000.0

                # 지게차/배경 구조물 제외 판정
                is_hydrant_or_vehicle = (depth_std < 0.04) or (w > 200 and depth_std < 0.08)

                if is_hydrant_or_vehicle:
                    # 지게차/소화전 등 구조물로 판단하여 Ignored
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 255), 1)
                    cv2.putText(cv_image, 'Object (Ignored)', (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    # 불꽃 탐지
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(cv_image, f'FIRE DETECTED', (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    fire_detected = True

        # ==========================================================
        # 6. 화재 감지 여부에 따른 모니터링 창 제어 [수정된 부분]
        # ==========================================================
        if fire_detected:
            # 화재가 감지되었을 때만 창을 생성/갱신
            cv2.imshow("Fire Detection Monitor", cv_image)
            cv2.waitKey(1)
            self.window_open = True
            self.get_logger().warn('FIRE DETECTED!')
        else:
            # 화재가 감지되지 않고 이전에 창이 열려 있었다면 창을 파괴(닫기)
            if self.window_open:
                cv2.destroyAllWindows()
                self.window_open = False

        # ==========================================================
        # 7. 결과 영상 및 알람 Publish
        # !! 발행만 하고 사용처가 없음
        # ==========================================================
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = rgb_msg.header
            self.image_pub.publish(processed_msg)
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

        # 화재 Alarm Publish
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
        cv2.destroyAllWindows()  # 노드 종료 시 남아있는 창 모두 닫기
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()