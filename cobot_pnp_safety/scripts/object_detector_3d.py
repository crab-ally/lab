#!/usr/bin/env python3
"""
3D RANSAC + DBSCAN Object Detector for Franka Emika Panda

Input:
  - /camera/depth/image_raw
  - /camera/camera_info
  - /camera/segmentation/image_raw

TF:
  - ceiling_camera_optical_frame -> link0

Output:
  - /target_object_pose
  - /detected_objects_markers
  - /object_pointcloud
"""

import math,cv2,numpy as np
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image,CameraInfo,PointCloud2,PointField
from tf2_ros import Buffer,TransformListener
import tf2_geometry_msgs

POSE_LOG_INTERVAL=1.0

class Ransac3DObjectDetector(Node):
    def __init__(self):
        super().__init__("ransac_3d_object_detector")
        self.camera_frame="ceiling_camera_optical_frame"
        self.target_frame="link0"
        self.depth=None
        self.seg=None
        self.depth_stamp=None
        self.fx,self.fy,self.cx,self.cy=432.97,432.97,319.5,239.5
        self.stride=2
        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)
        self.pose_pub=self.create_publisher(PoseStamped,"/target_object_pose",10)
        self.marker_pub=self.create_publisher(MarkerArray,"/detected_objects_markers",10)
        self.cloud_pub=self.create_publisher(PointCloud2,"/object_pointcloud",10)
        self.create_subscription(Image,"/camera/depth/image_raw",self.depth_callback,10)
        self.create_subscription(Image,"/camera/segmentation/image_raw",self.seg_callback,10)
        self.create_subscription(CameraInfo,"/camera/depth/camera_info",self.info_callback,10)
        self.last_pose_log_time=0.0
        self.create_timer(0.1,self.process)
        self.get_logger().info("3D RANSAC + DBSCAN Object Detector started.")

    def depth_callback(self,msg):
        if msg.encoding=="32FC1":
            self.depth=np.frombuffer(msg.data,dtype=np.float32).reshape(msg.height,msg.width).copy()
        elif msg.encoding=="16UC1":
            self.depth=np.frombuffer(msg.data,dtype=np.uint16).reshape(msg.height,msg.width).astype(np.float32)/1000.0
        else:return
        self.depth_stamp=msg.header.stamp

    def seg_callback(self,msg):
        if msg.encoding!="32SC1":return
        self.seg=np.frombuffer(msg.data,dtype=np.int32).reshape(msg.height,msg.width).copy()

    def info_callback(self,msg):
        if msg.k[0]>0 and msg.k[4]>0:self.fx,self.fy,self.cx,self.cy=msg.k[0],msg.k[4],msg.k[2],msg.k[5]

    def depth_to_points(self):
        d=self.depth[::self.stride,::self.stride]
        s=self.seg[::self.stride,::self.stride]
        h,w=d.shape
        u,v=np.meshgrid(np.arange(0,w*self.stride,self.stride),np.arange(0,h*self.stride,self.stride))
        d,s,u,v=d.reshape(-1),s.reshape(-1),u.reshape(-1),v.reshape(-1)
        valid=np.isfinite(d)&(d>0.01)&(d<50.0)&(s!=0)
        z=d[valid]
        x=(u[valid]-self.cx)*z/self.fx
        y=(v[valid]-self.cy)*z/self.fy
        return np.column_stack((x,y,z))

    def ransac_plane(self,points,threshold=0.015,iterations=100):
        if len(points)<50:return None
        best_mask,best_count=None,0
        for _ in range(iterations):
            p1,p2,p3=points[np.random.choice(len(points),3,replace=False)]
            n=np.cross(p2-p1,p3-p1)
            norm=np.linalg.norm(n)
            if norm<1e-6:continue
            n/=norm
            d=-np.dot(n,p1)
            mask=np.abs(points@n+d)<threshold
            count=np.sum(mask)
            if count>best_count:best_count,best_mask=count,mask
        return None if best_mask is None else points[~best_mask]

    def compute_cluster_yaw_quaternion(self,cluster_points):
        """
        카메라 프레임의 X-Y 평면 상에서 물체의 2D minAreaRect를 구하고 쿼터니언을 반환합니다.
        """
        pts_2d=cluster_points[:,:2].astype(np.float32)
        rect=cv2.minAreaRect(pts_2d)
        (w,h),angle=rect[1],rect[2]
        if w<h:angle+=90.0
        yaw_rad=math.radians(angle)

        # Z축 회전(Yaw)에 대한 Quaternion 생성
        return [0.0,0.0,math.sin(yaw_rad/2.0),math.cos(yaw_rad/2.0)]

    def transform_pose(self,camera_pose_stamped):
        """
        PoseStamped: camera_frame → target_frame(link0)
        """
        try:
            # 최신 tf 사용 (인식 후 로봇 이동)
            tf=self.tf_buffer.lookup_transform(self.target_frame,self.camera_frame,rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=0.2))
            return tf2_geometry_msgs.do_transform_pose_stamped(camera_pose_stamped,tf)
        except Exception as e:
            self.get_logger().error(f"TF Transform failed: {e}")
            return None

    def publish_pointcloud(self,points):
        msg=PointCloud2()
        msg.header.stamp=self.depth_stamp if self.depth_stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id=self.camera_frame
        msg.height=1
        msg.width=len(points)
        msg.fields=[
            PointField(name="x",offset=0,datatype=PointField.FLOAT32,count=1),
            PointField(name="y",offset=4,datatype=PointField.FLOAT32,count=1),
            PointField(name="z",offset=8,datatype=PointField.FLOAT32,count=1)]
        msg.is_bigendian=False
        msg.point_step=12
        msg.row_step=12*len(points)
        msg.is_dense=True
        msg.data=points.astype(np.float32).tobytes()
        self.cloud_pub.publish(msg)

    def process(self):
        if self.depth is None or self.seg is None:return
        if self.depth.shape!=self.seg.shape:return

        points=self.depth_to_points()
        if len(points)<100:return

        object_points=self.ransac_plane(points)
        if object_points is None or len(object_points)<20:return

        self.publish_pointcloud(object_points)

        labels=DBSCAN(eps=0.06,min_samples=8).fit_predict(object_points)
        clusters=[object_points[labels==i] for i in set(labels) if i>=0 and np.sum(labels==i)>=20]
        if not clusters:return

        markers=MarkerArray()
        stamp=self.depth_stamp

        # 이전 Marker 삭제
        delete=Marker()
        delete.header.frame_id=self.target_frame
        delete.header.stamp=stamp
        delete.action=Marker.DELETEALL
        markers.markers.append(delete)

        published_count=0

        for i,cluster in enumerate(clusters):
            # 1. 카메라 프레임 기준 중심점 및 크기 계산
            center=(cluster.min(axis=0)+cluster.max(axis=0))/2.0
            size=cluster.max(axis=0)-cluster.min(axis=0)

            # 2. 카메라 프레임 기준 Orientation 계산
            q_cam=self.compute_cluster_yaw_quaternion(cluster)

            # 3. 카메라 프레임 기준 PoseStamped 메시지 생성
            cam_pose=PoseStamped()
            cam_pose.header.frame_id=self.camera_frame
            cam_pose.header.stamp=stamp
            cam_pose.pose.position.x=float(center[0])
            cam_pose.pose.position.y=float(center[1])
            cam_pose.pose.position.z=float(center[2])
            cam_pose.pose.orientation.x=q_cam[0]
            cam_pose.pose.orientation.y=q_cam[1]
            cam_pose.pose.orientation.z=q_cam[2]
            cam_pose.pose.orientation.w=q_cam[3]

            # 4. [좌표계 변환] 카메라 프레임 Pose -> link0 기준 Pose로 변환
            target_pose=self.transform_pose(cam_pose)
            if target_pose is None:continue

            px,py,pz=target_pose.pose.position.x,target_pose.pose.position.y,target_pose.pose.position.z

            # 첫 번째 객체에 대해서만 PoseStamped 발행
            if published_count==0:
                self.pose_pub.publish(target_pose)
                now=self.get_clock().now().nanoseconds*1e-9
                if now-self.last_pose_log_time>=POSE_LOG_INTERVAL:
                    self.get_logger().info(f"Target pose (in {self.target_frame}): pos=({px:.3f}, {py:.3f}, {pz:.3f})")
                    self.last_pose_log_time=now

            # 5. RViz 시각화를 위한 Marker 설정 (변환된 target_pose 활용)
            marker=Marker()
            marker.header.frame_id=self.target_frame
            marker.header.stamp=stamp
            marker.ns="objects"
            marker.id=i
            marker.type=Marker.CUBE
            marker.action=Marker.ADD
            marker.pose=target_pose.pose
            marker.scale.x=max(float(size[0]),0.04)
            marker.scale.y=max(float(size[1]),0.04)
            marker.scale.z=max(float(size[2]),0.04)
            marker.color.r=1.0
            marker.color.g=0.0
            marker.color.b=0.0
            marker.color.a=0.6
            marker.lifetime.sec=0
            marker.lifetime.nanosec=0
            markers.markers.append(marker)
            published_count+=1

        if published_count>0:self.marker_pub.publish(markers)

def main():
    rclpy.init()
    node=Ransac3DObjectDetector()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()