#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class YoloDetectorNode(Node):
    """
    Subscribe: /camera/image_raw
    Publish: /yolo/detections (BBox, Class ID, Conf)
    """
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()

        model_path = '/workspace/models/ppe_forklift_yolov8n/best.pt'
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.conf_thresh = 0.45

        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_det = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.get_logger().info('YOLO Detector Node started.')

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        results = self.model(cv_img, conf=self.conf_thresh, verbose=False)
        det_array_msg = Detection2DArray()
        det_array_msg.header = msg.header  # 카메라 타임스탬프 동기화 유지

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                det = Detection2D()
                det.header = msg.header
                
                # BBox 중심 및 크기 저장 (Center X, Center Y, Width, Height)
                det.bbox.center.position.x = (x1 + x2) / 2.0
                det.bbox.center.position.y = (y1 + y2) / 2.0
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1

                # Class ID 및 Confidence 저장
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                det_array_msg.detections.append(det)

        self.pub_det.publish(det_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()