#!/usr/bin/env python3
"""
입력:
    /camera/image_raw
    /camera/depth/image_raw
    /camera/depth/camera_info

출력:
    /camera/fall_detection/image
    /fall_alarm (JSON)
"""

import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import json
import math
from typing import Optional,Tuple

import cv2
import numpy as np
import rclpy
import tf2_ros

from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point
from builtin_interfaces.msg import Time
from ultralytics import YOLO


class FallDetectionNode(Node):

    def __init__(self):
        super().__init__('fall_detection_node')
        self.bridge=CvBridge()

        # ========================================================
        # YOLO Pose Model
        # ========================================================

        self.get_logger().info('Loading YOLOv8-pose model...')
        self.model=YOLO('/workspace/models/fall_yolov8n_pose/best.pt')
        self.get_logger().info('YOLOv8-pose model loaded successfully.')

        # ========================================================
        # Thresholds
        # ========================================================

        self.CONF_THRESHOLD=0.5
        self.KEYPOINT_CONF_THRESHOLD=0.5
        self.HORIZONTAL_ANGLE_THRESHOLD=35.0

        # ========================================================
        # Coordinate Parameters
        # ========================================================

        self.declare_parameter('target_frame','base_link')
        self.target_frame=self.get_parameter('target_frame').get_parameter_value().string_value

        self.declare_parameter('map_frame','map')
        self.map_frame=self.get_parameter('map_frame').get_parameter_value().string_value

        self.declare_parameter('min_depth',0.2)
        self.min_depth=self.get_parameter('min_depth').get_parameter_value().double_value

        self.declare_parameter('max_depth',8.0)
        self.max_depth=self.get_parameter('max_depth').get_parameter_value().double_value

        # ========================================================
        # TF2
        # ========================================================

        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer,self)

        # ========================================================
        # Camera Intrinsics
        # ========================================================

        self.fx=600.0
        self.fy=600.0
        self.cx=320.0
        self.cy=240.0
        self.camera_info_received=False
        self.camera_frame_id='camera_optical_frame'

        # ========================================================
        # Depth
        # ========================================================

        self.latest_depth_img:Optional[np.ndarray]=None
        self.latest_depth_encoding='16UC1'
        self.latest_depth_stamp=0.0
        self.latest_depth_frame='camera_optical_frame'

        # ========================================================
        # ROS Subscribers / Publishers
        # ========================================================

        self.image_sub=self.create_subscription(Image,'/camera/image_raw',self.image_callback,10)
        self.depth_sub=self.create_subscription(Image,'/camera/depth/image_raw',self.depth_callback,10)
        self.camera_info_sub=self.create_subscription(CameraInfo,'/camera/depth/camera_info',self.camera_info_callback,10)

        self.image_pub=self.create_publisher(Image,'/camera/fall_detection/image',10)
        self.alert_pub=self.create_publisher(String,'/fall_alarm',10)

        # ========================================================
        # Tracking
        # ========================================================

        self.tracking_history={}
        self.TRACKING_WARMUP_FRAMES=10
        self.MAX_MISSED_FRAMES=5
        self.missed_frames={}

        # ========================================================
        # Initial State Voting
        # ========================================================

        self.initial_fall_votes={}
        self.INITIAL_FALL_RATIO=0.5

        # ========================================================
        # Fall Detection
        # ========================================================

        self.fall_history={}
        self.FALL_CANDIDATE_FRAMES=20

        # ========================================================
        # Track State
        # ========================================================

        self.track_states={}
        self.get_logger().info('Fall Detection Node has been started.')

    # ============================================================
    # Timestamp
    # ============================================================

    @staticmethod
    def _stamp_to_float(stamp:Time)->float:
        return float(stamp.sec)+float(stamp.nanosec)*1e-9

    # ============================================================
    # Depth Callback
    # ============================================================

    def depth_callback(self,msg):
        try:
            if msg.encoding in ['16UC1','mono16']:
                self.latest_depth_img=self.bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
                self.latest_depth_encoding='16UC1'
            elif msg.encoding=='32FC1':
                self.latest_depth_img=self.bridge.imgmsg_to_cv2(msg,desired_encoding='32FC1')
                self.latest_depth_encoding='32FC1'
            else:
                self.get_logger().warning(f'Unsupported depth encoding: {msg.encoding}')
                return

            self.latest_depth_stamp=self._stamp_to_float(msg.header.stamp)
            self.latest_depth_frame=msg.header.frame_id

        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    # ============================================================
    # Camera Info Callback
    # ============================================================

    def camera_info_callback(self,msg):
        if msg.k[0]>0:
            self.fx=msg.k[0]

        if msg.k[4]>0:
            self.fy=msg.k[4]

        self.cx=msg.k[2]
        self.cy=msg.k[5]

        if msg.header.frame_id:
            self.camera_frame_id=msg.header.frame_id

        if not self.camera_info_received:
            self.camera_info_received=True
            self.get_logger().info('Camera Intrinsics Loaded')

    # ============================================================
    # TF
    # ============================================================

    def _get_tf(self,target_frame,source_frame,stamp):
        try:
            tf_time=rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(target_frame,source_frame,tf_time,timeout=rclpy.duration.Duration(seconds=0.05))
        except (tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException):
            try:
                return self.tf_buffer.lookup_transform(target_frame,source_frame,rclpy.time.Time())
            except Exception as e:
                self.get_logger().debug(f'TF unavailable ({source_frame} -> {target_frame}): {e}')
                return None

    # ============================================================
    # Depth
    # ============================================================

    def _get_depth_at_bbox_center(self,x1,y1,x2,y2):
        if self.latest_depth_img is None:
            return None

        h_img,w_img=self.latest_depth_img.shape[:2]

        x1=max(0,min(w_img-1,x1))
        x2=max(0,min(w_img,x2))
        y1=max(0,min(h_img-1,y1))
        y2=max(0,min(h_img,y2))

        if x2<=x1 or y2<=y1:
            return None

        cx_box=(x1+x2)/2.0
        cy_box=(y1+y2)/2.0
        box_w=x2-x1
        box_h=y2-y1

        rx1=max(0,int(cx_box-box_w*0.2))
        rx2=min(w_img,int(cx_box+box_w*0.2))
        ry1=max(0,int(cy_box-box_h*0.2))
        ry2=min(h_img,int(cy_box+box_h*0.2))

        if rx2<=rx1 or ry2<=ry1:
            return None

        depth_roi=self.latest_depth_img[ry1:ry2,rx1:rx2]

        if self.latest_depth_encoding=='16UC1':
            valid_depths=depth_roi[depth_roi>0].astype(np.float32)/1000.0
        else:
            valid_depths=depth_roi[np.isfinite(depth_roi)&(depth_roi>0.1)]

        valid_depths=valid_depths[(valid_depths>=self.min_depth)&(valid_depths<=self.max_depth)]

        if len(valid_depths)<10:
            return None

        return float(np.median(valid_depths))

    # ============================================================
    # Camera Pixel → 3D
    # ============================================================

    def _calculate_camera_3d(self,u,v,depth_m):
        if self.fx<=0.0 or self.fy<=0.0 or depth_m is None:
            return None

        x_cam=(u-self.cx)*depth_m/self.fx
        y_cam=(v-self.cy)*depth_m/self.fy
        z_cam=depth_m

        return x_cam,y_cam,z_cam

    # ============================================================
    # Camera → base_link → map
    # ============================================================

    def _calculate_frame_coordinates(self,u,v,depth_m,stamp):
        camera_position=self._calculate_camera_3d(u,v,depth_m)

        if camera_position is None:
            return None,None

        cam_x,cam_y,cam_z=camera_position

        p_cam=PointStamped()
        p_cam.header.frame_id=self.camera_frame_id
        p_cam.header.stamp=stamp
        p_cam.point.x=float(cam_x)
        p_cam.point.y=float(cam_y)
        p_cam.point.z=float(cam_z)

        tf_camera_to_base=self._get_tf(self.target_frame,self.camera_frame_id,stamp)

        if tf_camera_to_base is None:
            return None,None

        try:
            p_base=do_transform_point(p_cam,tf_camera_to_base)
        except Exception as e:
            self.get_logger().debug(f'Camera -> {self.target_frame} transform failed: {e}')
            return None,None

        base_position=[round(float(p_base.point.x),2),round(float(p_base.point.y),2),round(float(p_base.point.z),2)]

        map_position=None

        tf_base_to_map=self._get_tf(self.map_frame,self.target_frame,stamp)

        if tf_base_to_map is not None:
            try:
                p_map=do_transform_point(p_base,tf_base_to_map)
                map_position=[round(float(p_map.point.x),2),round(float(p_map.point.y),2),round(float(p_map.point.z),2)]
            except Exception as e:
                self.get_logger().debug(f'{self.target_frame} -> {self.map_frame} transform failed: {e}')

        return base_position,map_position

    # ============================================================
    # Keypoint Center
    # ============================================================

    def get_center(self,kpt,ids):
        points=[]
        num_kpts=len(kpt)

        for idx in ids:
            if idx>=num_kpts:
                continue

            x,y,conf=kpt[idx]

            if conf>=self.KEYPOINT_CONF_THRESHOLD:
                points.append((float(x),float(y)))

        if not points:
            return None

        x=sum(p[0] for p in points)/len(points)
        y=sum(p[1] for p in points)/len(points)

        return x,y

    # ============================================================
    # Keypoint Geometry
    # ============================================================

    def get_geometry_points(self,kpt):
        num_kpts=len(kpt)

        if num_kpts==12:
            shoulder_ids=[1,2]
            hip_ids=[7]
            knee_ids=[8,9]
        elif num_kpts>=17:
            shoulder_ids=[5,6]
            hip_ids=[11,12]
            knee_ids=[13,14]
        else:
            return None

        shoulder=self.get_center(kpt,shoulder_ids)
        hip=self.get_center(kpt,hip_ids)
        knee=self.get_center(kpt,knee_ids)

        if shoulder is None or hip is None or knee is None:
            return None

        return shoulder,hip,knee

    # ============================================================
    # Line Angle
    # ============================================================

    def calculate_line_angle(self,p1,p2):
        dx=p2[0]-p1[0]
        dy=p2[1]-p1[1]

        if abs(dx)<1e-6 and abs(dy)<1e-6:
            return None

        angle=math.degrees(math.atan2(dy,dx))
        angle=abs(angle)

        if angle>90.0:
            angle=180.0-angle

        return angle

    # ============================================================
    # Geometry Validation
    # ============================================================

    def calculate_geometry(self,kpt):
        points=self.get_geometry_points(kpt)

        if points is None:
            return False,None,None,None

        shoulder,hip,knee=points
        shoulder_hip_angle=self.calculate_line_angle(shoulder,hip)
        hip_knee_angle=self.calculate_line_angle(hip,knee)

        if shoulder_hip_angle is None or hip_knee_angle is None:
            return False,shoulder_hip_angle,hip_knee_angle,points

        geometry_fall=shoulder_hip_angle<=self.HORIZONTAL_ANGLE_THRESHOLD and hip_knee_angle<=self.HORIZONTAL_ANGLE_THRESHOLD

        return geometry_fall,shoulder_hip_angle,hip_knee_angle,points

    # ============================================================
    # Fall Alarm
    # ============================================================

    def publish_fall_alarm(self,track_id,confidence,isfallen,stamp,position=None,position_map=None,class_name='person'):
        data={
            'track_id':track_id,
            'class_name':class_name,
            'confidence':confidence,
            'isfallen':isfallen,
            'stamp':stamp,
            'position':position,
            'position_map':position_map
        }
        msg=String()
        msg.data=json.dumps(data,ensure_ascii=False)
        self.alert_pub.publish(msg)

    # ============================================================
    # Geometry Visualization
    # ============================================================

    def draw_geometry(self,image,points,color):
        if points is None:
            return

        shoulder,hip,knee=points
        s=(int(shoulder[0]),int(shoulder[1]))
        h=(int(hip[0]),int(hip[1]))
        k=(int(knee[0]),int(knee[1]))

        cv2.line(image,s,h,color,3)
        cv2.line(image,h,k,color,3)
        cv2.circle(image,s,6,color,-1)
        cv2.circle(image,h,6,color,-1)
        cv2.circle(image,k,6,color,-1)

    # ============================================================
    # Image Callback
    # ============================================================

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        stamp=self._stamp_to_float(msg.header.stamp)

        results=self.model.track(cv_image,persist=True,conf=self.CONF_THRESHOLD,verbose=False)

        current_track_ids=set()
        person_detected_global=False

        if results and len(results)>0:
            result=results[0]
            boxes=result.boxes
            keypoints=result.keypoints

            if boxes is not None and keypoints is not None:
                track_ids=boxes.id.int().cpu().tolist() if boxes.id is not None else []
                classes=boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
                xyxy=boxes.xyxy.cpu().numpy()
                confidences=boxes.conf.cpu().numpy() if boxes.conf is not None else []
                kpts=keypoints.data.cpu().numpy()

                for i in range(len(xyxy)):
                    x1,y1,x2,y2=map(int,xyxy[i])
                    track_id=track_ids[i] if i<len(track_ids) else -1

                    if track_id==-1:
                        continue

                    confidence=float(confidences[i]) if i<len(confidences) else 0.0

                    if confidence<self.CONF_THRESHOLD:
                        continue

                    current_track_ids.add(track_id)
                    person_detected_global=True
                    self.missed_frames[track_id]=0

                    kpt=kpts[i]
                    cls_id=classes[i] if i<len(classes) else 0

                    yolo_falling=cls_id==0
                    geometry_fall,shoulder_hip_angle,hip_knee_angle,geometry_points=self.calculate_geometry(kpt)
                    is_falling=yolo_falling and geometry_fall

                    self.tracking_history[track_id]=self.tracking_history.get(track_id,0)+1
                    tracking_frames=self.tracking_history[track_id]

                    if track_id not in self.initial_fall_votes:
                        self.initial_fall_votes[track_id]=0

                    if track_id not in self.fall_history:
                        self.fall_history[track_id]=0

                    if track_id not in self.track_states:
                        self.track_states[track_id]='tracking'

                    if tracking_frames<=self.TRACKING_WARMUP_FRAMES:
                        if is_falling:
                            self.initial_fall_votes[track_id]+=1

                        color=(255,255,0)
                        label=f"ID:{track_id} TRACKING ({tracking_frames}/{self.TRACKING_WARMUP_FRAMES}) Conf:{confidence:.2f}"

                    elif self.track_states[track_id]=='tracking':
                        fall_votes=self.initial_fall_votes[track_id]
                        fall_ratio=fall_votes/float(self.TRACKING_WARMUP_FRAMES)

                        if fall_ratio>=self.INITIAL_FALL_RATIO:
                            self.track_states[track_id]='unsafe'
                            self.fall_history[track_id]=self.FALL_CANDIDATE_FRAMES
                            color=(0,0,255)
                            label=f"ID:{track_id} UNSAFE (Fall) Conf:{confidence:.2f}"
                        else:
                            self.track_states[track_id]='safe'
                            self.fall_history[track_id]=0
                            color=(0,255,0)
                            label=f"ID:{track_id} SAFE (Normal) Conf:{confidence:.2f}"

                    current_state=self.track_states[track_id]

                    # SAFE
                    if current_state=='safe':
                        if is_falling:
                            self.track_states[track_id]='fall_candidate'
                            self.fall_history[track_id]=1
                            color=(0,165,255)
                            label=f"ID:{track_id} FALL_CANDIDATE (1/{self.FALL_CANDIDATE_FRAMES}) Conf:{confidence:.2f}"
                        else:
                            self.fall_history[track_id]=0
                            color=(0,255,0)
                            label=f"ID:{track_id} SAFE Conf:{confidence:.2f}"

                    # FALL CANDIDATE
                    elif current_state=='fall_candidate':
                        if is_falling:
                            self.fall_history[track_id]+=1

                            if self.fall_history[track_id]>=self.FALL_CANDIDATE_FRAMES:
                                self.track_states[track_id]='unsafe'
                                self.fall_history[track_id]=self.FALL_CANDIDATE_FRAMES
                                color=(0,0,255)
                                label=f"ID:{track_id} UNSAFE (Fall) Conf:{confidence:.2f}"
                            else:
                                color=(0,165,255)
                                label=f"ID:{track_id} FALL_CANDIDATE ({self.fall_history[track_id]}/{self.FALL_CANDIDATE_FRAMES}) Conf:{confidence:.2f}"
                        else:
                            self.fall_history[track_id]=0
                            self.track_states[track_id]='safe'
                            color=(0,255,0)
                            label=f"ID:{track_id} SAFE Conf:{confidence:.2f}"

                    # UNSAFE
                    elif current_state=='unsafe':
                        if is_falling:
                            self.fall_history[track_id]=min(self.FALL_CANDIDATE_FRAMES,self.fall_history[track_id]+1)
                            color=(0,0,255)
                            label=f"ID:{track_id} UNSAFE (Fall) Conf:{confidence:.2f}"
                        else:
                            self.fall_history[track_id]=max(0,self.fall_history[track_id]-2)

                            if self.fall_history[track_id]==0:
                                self.track_states[track_id]='safe'
                                color=(0,255,0)
                                label=f"ID:{track_id} SAFE Conf:{confidence:.2f}"
                            else:
                                color=(0,0,255)
                                label=f"ID:{track_id} UNSAFE (Fall) Conf:{confidence:.2f}"

                    # ====================================================
                    # 최종 낙상 상태
                    # ====================================================

                    isfallen=self.track_states[track_id]=='unsafe'

                    # ====================================================
                    # 사람 위치 계산
                    # ====================================================

                    bbox_center_x=(x1+x2)/2.0
                    bbox_center_y=(y1+y2)/2.0
                    depth_m=self._get_depth_at_bbox_center(x1,y1,x2,y2)

                    position=None
                    position_map=None

                    if depth_m is not None:
                        position,position_map=self._calculate_frame_coordinates(bbox_center_x,bbox_center_y,depth_m,msg.header.stamp)

                    # ====================================================
                    # Fall Alarm
                    # ====================================================

                    self.publish_fall_alarm(track_id=track_id,confidence=confidence,isfallen=isfallen,stamp=stamp,position=position,position_map=position_map)

                    # ====================================================
                    # Bounding Box
                    # ====================================================

                    cv2.rectangle(cv_image,(x1,y1),(x2,y2),color,2)
                    cv2.putText(cv_image,label,(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

                    # ====================================================
                    # Geometry Information
                    # ====================================================

                    if shoulder_hip_angle is not None and hip_knee_angle is not None:
                        geometry_text=f"S-H:{shoulder_hip_angle:.1f} H-K:{hip_knee_angle:.1f}"
                        cv2.putText(cv_image,geometry_text,(x1,min(cv_image.shape[0]-10,y2+25)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                    # ====================================================
                    # Coordinate Information
                    # ====================================================

                    if position is not None:
                        base_text=f"BASE:{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}"
                        cv2.putText(cv_image,base_text,(x1,min(cv_image.shape[0]-35,y2+45)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

                    if position_map is not None:
                        map_text=f"MAP:{position_map[0]:.2f},{position_map[1]:.2f},{position_map[2]:.2f}"
                        cv2.putText(cv_image,map_text,(x1,min(cv_image.shape[0]-15,y2+65)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

                    # ====================================================
                    # Skeleton
                    # ====================================================

                    num_kpts=len(kpt)

                    if num_kpts==12:
                        skeleton_edges=[(1,3),(3,5),(2,4),(4,6),(1,2),(1,7),(2,7),(7,8),(8,10),(7,9),(9,11)]
                    else:
                        skeleton_edges=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

                    # ====================================================
                    # Keypoint 표시
                    # ====================================================

                    for pt_idx in range(num_kpts):
                        px,py,conf=kpt[pt_idx]

                        if conf>self.KEYPOINT_CONF_THRESHOLD:
                            cv2.circle(cv_image,(int(px),int(py)),4,color,-1)

                    # ====================================================
                    # Geometry 표시
                    # ====================================================

                    self.draw_geometry(cv_image,geometry_points,color)

                    # ====================================================
                    # Skeleton 연결
                    # ====================================================

                    for edge in skeleton_edges:
                        p1,p2=edge

                        if p1<num_kpts and p2<num_kpts:
                            x1_k,y1_k,conf1=kpt[p1]
                            x2_k,y2_k,conf2=kpt[p2]

                            if conf1>self.KEYPOINT_CONF_THRESHOLD and conf2>self.KEYPOINT_CONF_THRESHOLD:
                                cv2.line(cv_image,(int(x1_k),int(y1_k)),(int(x2_k),int(y2_k)),color,2)

        # ========================================================
        # Detection Loss / Track 유지
        # ========================================================

        all_track_ids=set(self.tracking_history.keys())

        for track_id in all_track_ids:
            if track_id in current_track_ids:
                self.missed_frames[track_id]=0
                continue

            self.missed_frames[track_id]=self.missed_frames.get(track_id,0)+1

            if self.missed_frames[track_id]<=self.MAX_MISSED_FRAMES:
                continue

            self.tracking_history.pop(track_id,None)
            self.initial_fall_votes.pop(track_id,None)
            self.fall_history.pop(track_id,None)
            self.track_states.pop(track_id,None)
            self.missed_frames.pop(track_id,None)

        # ========================================================
        # 결과 이미지 Publish
        # ========================================================

        if person_detected_global:
            try:
                processed_msg=self.bridge.cv2_to_imgmsg(cv_image,encoding='bgr8')
                processed_msg.header=msg.header
                self.image_pub.publish(processed_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node=FallDetectionNode()

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


if __name__=='__main__':
    main()