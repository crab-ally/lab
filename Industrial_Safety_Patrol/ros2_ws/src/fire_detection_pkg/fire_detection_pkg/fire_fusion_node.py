#!/usr/bin/env python3
"""
Node2: FireFusionNode

SUB:
    /fire_candidates
    /camera/image_raw
    /camera/depth/image_raw
    /camera/depth/camera_info
    /scan

PUB:
    /fire_tracks_3d
    /fire_alarm
    /fire_fusion/debug_markers
"""

import json,math,cv2
from collections import deque
from typing import List,Optional,Tuple
import numpy as np
import rclpy,tf2_ros,message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import Image,CameraInfo,LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker,MarkerArray
from builtin_interfaces.msg import Time

class FireFusionNode(Node):
    def __init__(self):
        super().__init__('fire_fusion_node')
        self.bridge=CvBridge()
        self.window_open=False

        self.declare_parameter('target_frame','base_link')
        self.target_frame=self.get_parameter('target_frame').value
        self.declare_parameter('map_frame','map')
        self.map_frame=self.get_parameter('map_frame').value
        self.declare_parameter('lidar_depth_tolerance',0.5)
        self.lidar_depth_tolerance=float(self.get_parameter('lidar_depth_tolerance').value)
        self.declare_parameter('min_depth',0.2)
        self.min_depth=float(self.get_parameter('min_depth').value)
        self.declare_parameter('max_depth',8.0)
        self.max_depth=float(self.get_parameter('max_depth').value)
        self.declare_parameter('sync_slop',0.1)
        self.sync_slop=float(self.get_parameter('sync_slop').value)
        self.declare_parameter('candidate_sync_tolerance',0.1)
        self.candidate_sync_tolerance=float(self.get_parameter('candidate_sync_tolerance').value)
        self.declare_parameter('lidar_sync_tolerance',0.15)
        self.lidar_sync_tolerance=float(self.get_parameter('lidar_sync_tolerance').value)

        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer,self)

        self.fx=600.0
        self.fy=600.0
        self.cx=320.0
        self.cy=240.0
        self.camera_info_received=False
        self.camera_frame_id='camera_color_optical_frame'

        self.sync_frames=deque(maxlen=20)
        self.scan_cache=deque(maxlen=30)
        self.latest_rgb_img=None
        self.latest_depth_img=None
        self.latest_depth_encoding='16UC1'
        self.latest_scan=None

        sensor_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)

        self.sub_candidates=self.create_subscription(String,'/fire_candidates',self._candidates_callback,10)

        self.rgb_sub=message_filters.Subscriber(self,Image,'/camera/image_raw',qos_profile=sensor_qos)
        self.depth_sub=message_filters.Subscriber(self,Image,'/camera/depth/image_raw',qos_profile=sensor_qos)
        self.rgb_depth_sync=message_filters.ApproximateTimeSynchronizer([self.rgb_sub,self.depth_sub],queue_size=10,slop=self.sync_slop)
        self.rgb_depth_sync.registerCallback(self._rgb_depth_callback)

        self.sub_info=self.create_subscription(CameraInfo,'/camera/depth/camera_info',self._camera_info_callback,10)
        self.sub_scan=self.create_subscription(LaserScan,'/scan',self._scan_callback,sensor_qos)

        self.pub_fire_tracks=self.create_publisher(String,'/fire_tracks_3d',10)
        self.pub_alarm=self.create_publisher(Bool,'/fire_alarm',10)
        self.pub_markers=self.create_publisher(MarkerArray,'/fire_fusion/debug_markers',10)

        self.get_logger().info(f'Fire Fusion Node started')

    @staticmethod
    def _stamp_to_float(stamp):
        return float(stamp.sec)+float(stamp.nanosec)*1e-9

    def _float_to_ros_time(self,value):
        try:
            value=float(value)
            sec=int(value)
            nanosec=int((value-sec)*1e9)
            if nanosec>=1000000000:
                sec+=1
                nanosec-=1000000000
            return Time(sec=sec,nanosec=nanosec)
        except Exception:
            return Time()

    def _camera_info_callback(self,msg:CameraInfo):
        if msg.k[0]>0:self.fx=msg.k[0]
        if msg.k[4]>0:self.fy=msg.k[4]
        self.cx=msg.k[2]
        self.cy=msg.k[5]
        if msg.header.frame_id:self.camera_frame_id=msg.header.frame_id
        if not self.camera_info_received:
            self.camera_info_received=True
            self.get_logger().info(f'CameraInfo loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}, frame={self.camera_frame_id}')

    def _rgb_depth_callback(self,rgb_msg:Image,depth_msg:Image):
        try:
            rgb_image=self.bridge.imgmsg_to_cv2(rgb_msg,desired_encoding='bgr8')
            if depth_msg.encoding in ['16UC1','mono16']:
                depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='passthrough')
                depth_encoding='16UC1'
            elif depth_msg.encoding=='32FC1':
                depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='32FC1')
                depth_encoding='32FC1'
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {depth_msg.encoding}')
                return
            rgb_stamp=self._stamp_to_float(rgb_msg.header.stamp)
            depth_stamp=self._stamp_to_float(depth_msg.header.stamp)
            if abs(rgb_stamp-depth_stamp)>self.sync_slop:return
            self.sync_frames.append({'stamp':rgb_stamp,'rgb':rgb_image,'depth':depth_image,'depth_encoding':depth_encoding,'rgb_header':rgb_msg.header,'depth_header':depth_msg.header})
        except Exception as e:
            self.get_logger().error(f'RGB/Depth synchronization error: {e}')

    def _scan_callback(self,msg:LaserScan):
        self.scan_cache.append({'stamp':self._stamp_to_float(msg.header.stamp),'scan':msg})

    def _get_synced_frame(self,target_stamp):
        if not self.sync_frames:return None
        best=min(self.sync_frames,key=lambda x:abs(x['stamp']-target_stamp))
        delta=abs(best['stamp']-target_stamp)
        if delta>self.candidate_sync_tolerance:
            self.get_logger().debug(f'No synchronized RGB/Depth frame: delta={delta:.3f}s')
            return None
        return best

    def _get_synced_scan(self,target_stamp)->Optional[LaserScan]:
        if not self.scan_cache:return None
        best=min(self.scan_cache,key=lambda x:abs(x['stamp']-target_stamp))
        delta=abs(best['stamp']-target_stamp)
        if delta>self.lidar_sync_tolerance:return None
        return best['scan']

    def _get_tf(self,target_frame,source_frame,stamp):
        try:
            tf_time=rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(target_frame,source_frame,tf_time)
        except(tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException):
            try:
                return self.tf_buffer.lookup_transform(target_frame,source_frame,rclpy.time.Time())
            except(tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException) as e:
                self.get_logger().debug(f'TF unavailable: {source_frame} -> {target_frame}: {e}')
                return None

    def _candidates_callback(self,msg:String):
        try:
            payload=json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Fire candidate JSON error: {e}')
            return

        header=payload.get('header',{})
        stamp_value=float(header.get('stamp',0.0))
        candidates=payload.get('candidates',[])

        if not candidates:
            self._publish_alarm(False)
            self._publish_tracks(stamp_value,[])
            self._publish_markers([])
            self._close_debug_window()
            return

        synced_frame=self._get_synced_frame(stamp_value)
        if synced_frame is None:return

        rgb_image=synced_frame['rgb']
        self.latest_rgb_img=rgb_image
        self.latest_depth_img=synced_frame['depth']
        self.latest_depth_encoding=synced_frame['depth_encoding']
        synced_scan=self._get_synced_scan(stamp_value)
        self.latest_scan=synced_scan

        stamp=self._float_to_ros_time(stamp_value)

        tf_camera_to_base=self._get_tf(self.target_frame,self.camera_frame_id,stamp)
        if tf_camera_to_base is None:
            self.get_logger().warn(f'Camera TF unavailable: {self.camera_frame_id} -> {self.target_frame}')
            return

        fire_tracks=[]

        for candidate in candidates:
            try:
                candidate_id=int(candidate['candidate_id'])
                bbox=candidate['bbox']
            except(KeyError,TypeError,ValueError):
                continue

            result=self._calculate_3d_from_depth(bbox)
            if result is None:continue

            cam_x,cam_y,cam_z=result
            lidar_distance=self._get_lidar_distance(cam_x,cam_z,synced_scan)

            fusion_distance=cam_z
            lidar_valid=False

            if lidar_distance is not None and abs(lidar_distance-cam_z)<=self.lidar_depth_tolerance:
                fusion_distance=0.7*cam_z+0.3*lidar_distance
                lidar_valid=True

            if cam_z<=0:continue

            scale=fusion_distance/cam_z
            cam_x*=scale
            cam_y*=scale
            cam_z=fusion_distance

            plane_valid=self._check_plane_geometry(bbox,cam_z)

            if synced_scan is not None and lidar_distance is not None and not lidar_valid:continue
            if not plane_valid:continue

            p_cam=PointStamped()
            p_cam.header.frame_id=self.camera_frame_id
            p_cam.header.stamp=stamp
            p_cam.point.x=cam_x
            p_cam.point.y=cam_y
            p_cam.point.z=cam_z

            try:
                p_base=do_transform_point(p_cam,tf_camera_to_base)
            except Exception as e:
                self.get_logger().warn(f'Camera -> base transform failed: {e}')
                continue

            base_x=p_base.point.x
            base_y=p_base.point.y
            base_z=p_base.point.z

            map_position=None
            tf_base_to_map=self._get_tf(self.map_frame,self.target_frame,stamp)

            if tf_base_to_map is not None:
                try:
                    p_map=do_transform_point(p_base,tf_base_to_map)
                    map_position=[round(float(p_map.point.x),2),round(float(p_map.point.y),2),round(float(p_map.point.z),2)]
                except Exception as e:
                    self.get_logger().debug(f'Base -> map transform failed: {e}')

            fire_track={
                'fire_id':candidate_id,
                'bbox':bbox,
                'position':[round(float(base_x),2),round(float(base_y),2),round(float(base_z),2)],
                'position_map':map_position,
                'distance':round(float(math.hypot(base_x,base_y)),2),
                'depth':round(float(cam_z),2),
                'lidar_distance':round(float(lidar_distance),2) if lidar_distance is not None else None,
                'lidar_valid':lidar_valid,
                'plane_valid':plane_valid,
                'temporal_hits':candidate.get('temporal_hits',0),
                'stamp':stamp_value
            }

            fire_tracks.append(fire_track)

        self._publish_tracks(stamp_value,fire_tracks)
        self._publish_alarm(len(fire_tracks)>0)
        self._publish_markers(fire_tracks)

        if fire_tracks:
            self._show_final_fire(rgb_image,fire_tracks)
            for fire in fire_tracks:
                self.get_logger().warn(f'FIRE DETECTED id={fire["fire_id"]} base={fire["position"]} map={fire["position_map"]} depth={fire["depth"]:.2f}m lidar={fire["lidar_distance"]}')
        else:
            self._close_debug_window()

    def _calculate_3d_from_depth(self,bbox:List[float])->Optional[Tuple[float,float,float]]:
        if self.latest_depth_img is None:return None

        h_img,w_img=self.latest_depth_img.shape[:2]
        xmin,ymin,xmax,ymax=map(int,bbox)

        xmin=max(0,min(w_img-1,xmin))
        xmax=max(0,min(w_img,xmax))
        ymin=max(0,min(h_img-1,ymin))
        ymax=max(0,min(h_img,ymax))

        if xmax<=xmin or ymax<=ymin:return None

        cx_box=(xmin+xmax)/2.0
        cy_box=(ymin+ymax)/2.0
        box_w=xmax-xmin
        box_h=ymax-ymin

        rx1=max(0,int(cx_box-box_w*0.25))
        rx2=min(w_img,int(cx_box+box_w*0.25))
        ry1=max(0,int(cy_box-box_h*0.25))
        ry2=min(h_img,int(cy_box+box_h*0.25))

        if rx2<=rx1 or ry2<=ry1:return None

        depth_roi=self.latest_depth_img[ry1:ry2,rx1:rx2]

        if self.latest_depth_encoding=='16UC1':
            valid_depths=depth_roi[depth_roi>0].astype(np.float32)/1000.0
        else:
            valid_depths=depth_roi[np.isfinite(depth_roi)&(depth_roi>0.1)]

        if len(valid_depths)<10:return None

        z_cam=float(np.median(valid_depths))

        if z_cam<self.min_depth or z_cam>self.max_depth:return None

        x_cam=(cx_box-self.cx)*z_cam/self.fx
        y_cam=(cy_box-self.cy)*z_cam/self.fy

        return x_cam,y_cam,z_cam

    def _get_lidar_distance(self,cam_x:float,cam_z:float,scan:Optional[LaserScan])->Optional[float]:
        if scan is None:return None

        angle_rad=-math.atan2(cam_x,cam_z)

        if angle_rad<scan.angle_min or angle_rad>scan.angle_max:return None

        idx=int((angle_rad-scan.angle_min)/scan.angle_increment)

        if not(0<=idx<len(scan.ranges)):return None

        scan_dist=scan.ranges[idx]

        if not np.isfinite(scan_dist):return None
        if not(scan.range_min<=scan_dist<=scan.range_max):return None

        return float(scan_dist)

    def _check_plane_geometry(self,bbox:List[float],representative_depth:float)->bool:
        if self.latest_depth_img is None:return False

        h_img,w_img=self.latest_depth_img.shape[:2]
        xmin,ymin,xmax,ymax=map(int,bbox)

        xmin=max(0,xmin)
        ymin=max(0,ymin)
        xmax=min(w_img,xmax)
        ymax=min(h_img,ymax)

        if xmax<=xmin or ymax<=ymin:return False

        roi=self.latest_depth_img[ymin:ymax,xmin:xmax]
        if roi.size==0:return False

        step_y=max(1,roi.shape[0]//15)
        step_x=max(1,roi.shape[1]//15)

        ys,xs=np.mgrid[ymin:ymax:step_y,xmin:xmax:step_x]
        xs=xs.flatten()
        ys=ys.flatten()

        if len(xs)<10:return True

        depth_values=[]

        for px,py in zip(xs,ys):
            depth=self.latest_depth_img[py,px]
            if self.latest_depth_encoding=='16UC1':
                depth=float(depth)/1000.0
            else:
                depth=float(depth)

            if not np.isfinite(depth) or depth<self.min_depth or depth>self.max_depth:continue

            depth_values.append((float(px),float(py),depth))

        if len(depth_values)<10:return True

        points=[]

        for px,py,z in depth_values:
            x=(px-self.cx)*z/self.fx
            y=(py-self.cy)*z/self.fy
            points.append([x,y,z])

        points=np.asarray(points,dtype=np.float64)
        center=np.mean(points,axis=0)
        centered=points-center

        try:
            _,_,vh=np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            return True

        normal=vh[-1]
        residuals=np.abs(centered@normal)

        median_residual=float(np.median(residuals))
        p95_residual=float(np.percentile(residuals,95))

        plane_like=median_residual<0.015 and p95_residual<0.05

        return not plane_like

    def _show_final_fire(self,image,fires):
        if image is None:return

        display=image.copy()

        for fire in fires:
            x1,y1,x2,y2=map(int,fire['bbox'])

            cv2.rectangle(display,(x1,y1),(x2,y2),(0,0,255),3)

            label=f'FINAL FIRE ID:{fire["fire_id"]} D:{fire["depth"]:.2f}m'

            cv2.putText(display,label,(x1,max(y1-10,25)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.imshow('Final Fire Detection',display)
        cv2.waitKey(1)
        self.window_open=True

    def _close_debug_window(self):
        if not self.window_open:return

        try:
            cv2.destroyWindow('Final Fire Detection')
            cv2.waitKey(1)
        except Exception:
            pass

        self.window_open=False

    def _publish_tracks(self,stamp,fires):
        output={
            'header':{
                'stamp':stamp,
                'frame_id':self.target_frame,
                'map_frame':self.map_frame
            },
            'fire_detected':len(fires)>0,
            'fire_count':len(fires),
            'fires':fires
        }

        msg=String()
        msg.data=json.dumps(output,ensure_ascii=False)
        self.pub_fire_tracks.publish(msg)

    def _publish_alarm(self,state):
        msg=Bool()
        msg.data=bool(state)
        self.pub_alarm.publish(msg)

    def _publish_markers(self,fires):
        marker_array=MarkerArray()

        for fire in fires:
            marker=Marker()
            marker.header.frame_id=self.target_frame
            marker.header.stamp=self.get_clock().now().to_msg()
            marker.ns='fire_tracks'
            marker.id=int(fire['fire_id'])
            marker.type=Marker.SPHERE
            marker.action=Marker.ADD

            x,y,z=fire['position']

            marker.pose.position.x=x
            marker.pose.position.y=y
            marker.pose.position.z=z
            marker.pose.orientation.w=1.0

            marker.scale.x=0.5
            marker.scale.y=0.5
            marker.scale.z=0.5

            marker.color.r=1.0
            marker.color.g=0.0
            marker.color.b=0.0
            marker.color.a=0.9

            marker_array.markers.append(marker)

        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node=FireFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Fire Fusion Node stopped.')
    finally:
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()