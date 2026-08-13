#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2DArray
import message_filters

class PPEDetectionNode(Node):
    """
    Subscribe:
        /camera/image_raw
        /yolo/detections
        /forklift_1/pose
        /forklift_2/pose
    Publish:
        /camera/ppe_detection/image
        /ppe_alarm
        /forklift_1/cmd_vel
        /forklift_2/cmd_vel
    """
    def __init__(self):
        super().__init__('ppe_detection_node')

        self.bridge = CvBridge()

        self.CLASS_PERSON = 0
        self.CLASS_HELMET = 1
        self.CLASS_VEST = 2
        self.CLASS_FORKLIFT = 3

        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

        # 원본 이미지와 YOLO 탐지 결과 동기화 구독
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.det_sub = message_filters.Subscriber(self, Detection2DArray, '/yolo/detections')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.det_sub], queue_size=10, slop=0.12
        )
        self.sync.registerCallback(self.ppe_callback)

        self.image_pub = self.create_publisher(
            Image,
            '/camera/ppe_detection/image',
            10
        )

        self.alert_pub = self.create_publisher(
            Bool,
            '/ppe_alarm',
            10
        )

        self.get_logger().info('PPE Detection Node has been started.')

        # Forklift Control Initialization
        self.fl1_pose = None
        self.fl2_pose = None
        
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
        
        self.fl1_pose_sub = self.create_subscription(PoseStamped, '/forklift_1/pose', self.fl1_pose_callback, 10)
        self.fl2_pose_sub = self.create_subscription(PoseStamped, '/forklift_2/pose', self.fl2_pose_callback, 10)
        
        self.fl1_cmd_pub = self.create_publisher(Twist, '/forklift_1/cmd_vel', 10)
        self.fl2_cmd_pub = self.create_publisher(Twist, '/forklift_2/cmd_vel', 10)
        
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
            dir_vec = diff / dist
            twist.linear.x = float(dir_vec[0] * self.fl_speed)
            twist.linear.y = float(dir_vec[1] * self.fl_speed)
            
        return twist, wp_idx

    def control_timer_callback(self):
        fl1_twist, self.fl1_wp_idx = self._compute_forklift_twist(self.fl1_pose, self.fl1_waypoints, self.fl1_wp_idx)
        self.fl1_cmd_pub.publish(fl1_twist)
        
        fl2_twist, self.fl2_wp_idx = self._compute_forklift_twist(self.fl2_pose, self.fl2_waypoints, self.fl2_wp_idx)
        self.fl2_cmd_pub.publish(fl2_twist)

    def ppe_callback(self, msg: Image, det_msg: Detection2DArray):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        unsafe_detected = False
        bbs = []
        helmets = []
        vests = []

        for det in det_msg.detections:
            if not det.results:
                continue
            cls_id = int(det.results[0].hypothesis.class_id)
            conf = float(det.results[0].hypothesis.score)

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)
            bbox = (x1, y1, x2, y2)

            if cls_id == self.CLASS_PERSON or cls_id == self.CLASS_FORKLIFT:
                bbs.append(([x1, y1, int(w), int(h)], conf, cls_id))
            elif cls_id == self.CLASS_HELMET:
                helmets.append(bbox)
            elif cls_id == self.CLASS_VEST:
                vests.append(bbox)

        tracks = self.tracker.update_tracks(bbs, frame=cv_image)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            px1, py1, px2, py2 = map(int, ltrb)
            cls_id = track.get_det_class()
            
            if isinstance(cls_id, str):
                cls_id = int(cls_id)
                
            if cls_id == self.CLASS_PERSON:
                has_helmet = False
                has_vest = False

                for (hx1, hy1, hx2, hy2) in helmets:
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0
                    if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                        has_helmet = True
                        break

                for (vx1, vy1, vx2, vy2) in vests:
                    vcx = (vx1 + vx2) / 2.0
                    vcy = (vy1 + vy2) / 2.0
                    if px1 <= vcx <= px2 and py1 <= vcy <= py2:
                        has_vest = True
                        break

                is_safe = has_helmet and has_vest

                if not is_safe:
                    unsafe_detected = True
                    color = (0, 0, 255)
                    label = f"Worker ID {track_id} (UNSAFE)"
                else:
                    color = (0, 255, 0)
                    label = f"Worker ID {track_id} (SAFE)"
                    
            elif cls_id == self.CLASS_FORKLIFT:
                color = (0, 165, 255)
                label = f"Forklift ID {track_id}"
            else:
                continue

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

        for (hx1, hy1, hx2, hy2) in helmets:
            cv2.rectangle(cv_image, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.putText(cv_image, "Helmet", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        for (vx1, vy1, vx2, vy2) in vests:
            cv2.rectangle(cv_image, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)
            cv2.putText(cv_image, "Vest", (vx1, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        try:
            processed_msg = self.bridge.cv2_to_imgmsg(
                cv_image,
                encoding='bgr8'
            )
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

        alert_msg = Bool()
        alert_msg.data = unsafe_detected
        self.alert_pub.publish(alert_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PPEDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('PPE Detection Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Exception in PPE Detection Node: {e}')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()