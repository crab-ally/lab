#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class FallDetectionNode(Node):
    """
    Pose 기반 쓰러짐 감지(Fall Detection) ROS2 Node

    입력:
        /camera/image_raw - 로봇 카메라 원본 이미지

    출력:
        /camera/fall_detection/image - Pose 및 쓰러짐 여부가 표시된 결과 이미지
        /fall_alarm - 쓰러짐 감지 여부 (T/F)
    """

    def __init__(self):
        super().__init__('fall_detection_node')
        self.bridge = CvBridge()
        
        self.get_logger().info('Loading YOLOv8-pose model (yolov8n-pose.pt)...')
        # 모델 경로 지정. 프로젝트 폴더가 /workspace 등이라면 절대경로 사용.
        # 여기서는 모델이 자동 다운로드/실행 디렉토리에 캐싱되도록 이름만 지정합니다.
        self.model = YOLO('yolov8n-pose.pt') 
        self.get_logger().info('YOLOv8-pose model loaded successfully.')
        
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
        
        # 사람별 쓰러짐 프레임 누적: {track_id: fall_frames}
        self.fall_history = {}
        # 쓰러짐으로 확정하기 위한 연속 프레임 임계값 (예: 20프레임)
        self.FALL_THRESHOLD_FRAMES = 20 
        
        self.get_logger().info('Fall Detection Node has been started.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # 사람 객체 추적 (트래킹을 통해 ID 부여)
        results = self.model.track(cv_image, persist=True, verbose=False)
        fall_detected_global = False
        
        current_track_ids = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy = boxes.xyxy.cpu().numpy()
                kpts = keypoints.data.cpu().numpy() # [N, 17, 3]
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    w = x2 - x1
                    h = y2 - y1
                    
                    track_id = track_ids[i] if i < len(track_ids) else -1
                    if track_id == -1:
                        continue
                        
                    current_track_ids.append(track_id)
                    kpt = kpts[i] # [17, 3] (x, y, conf)
                    
                    # ========================================================
                    # 1. Fall detection logic (쓰러짐 판정)
                    # ========================================================
                    
                    # Bounding box 가로/세로 비율 기준 판단
                    # 일반적으로 가로가 세로보다 길어질 때 쓰러진 것으로 간주
                    aspect_ratio = w / float(h),
                    
                    is_falling = False
                    if aspect_ratio > 1.2:
                        is_falling = True
                        
                    # ========================================================
                    # 2. Temporal Filtering (시간적 필터링)
                    # ========================================================
                    
                    if is_falling:
                        self.fall_history[track_id] = self.fall_history.get(track_id, 0) + 1
                    else:
                        # 정상 자세로 돌아오면 누적 카운트를 서서히 감소시킴
                        self.fall_history[track_id] = max(0, self.fall_history.get(track_id, 0) - 2)
                        
                    is_fallen_confirmed = self.fall_history[track_id] >= self.FALL_THRESHOLD_FRAMES
                    
                    if is_fallen_confirmed:
                        fall_detected_global = True
                        color = (0, 0, 255) # 쓰러짐: 빨간색
                        label = f"ID:{track_id} UNSAFE (Fall)"
                    else:
                        color = (0, 255, 0) # 정상: 초록색
                        label = f"ID:{track_id} SAFE (Normal)"
                        
                    # ========================================================
                    # 3. 시각화 (Bounding Box 및 스켈레톤)
                    # ========================================================
                    
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        cv_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
                    
                    # 스켈레톤 그리기
                    skeleton_edges = [
                        (5, 7), (7, 9), # Left arm
                        (6, 8), (8, 10), # Right arm
                        (5, 6), (11, 12), (5, 11), (6, 12), # Torso
                        (11, 13), (13, 15), # Left leg
                        (12, 14), (14, 16) # Right leg
                    ]
                    
                    # 관절 점
                    for pt_idx in range(17):
                        px, py, conf = kpt[pt_idx]
                        if conf > 0.5:
                            cv2.circle(cv_image, (int(px), int(py)), 4, color, -1)
                            
                    # 관절 선
                    for edge in skeleton_edges:
                        p1, p2 = edge
                        x1_k, y1_k, conf1 = kpt[p1]
                        x2_k, y2_k, conf2 = kpt[p2]
                        if conf1 > 0.5 and conf2 > 0.5:
                            cv2.line(cv_image, (int(x1_k), int(y1_k)), (int(x2_k), int(y2_k)), color, 2)
                            
        # ========================================================
        # 화면 출력 및 퍼블리시
        # ========================================================
        
        cv2.imshow("Fall Detection Monitor", cv_image)
        cv2.waitKey(1)
        
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")
            
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
