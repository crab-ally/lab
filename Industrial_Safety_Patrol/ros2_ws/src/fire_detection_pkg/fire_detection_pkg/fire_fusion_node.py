#!/usr/bin/env python3
"""
Node2: FireFusionNode

SUB:
    /fire_candidates
    /camera/depth/image_raw
    /camera/depth/camera_info
    /scan

PUB:
    /fire_tracks_3d
    /fire_alarm
    /fire_fusion/debug_markers
"""

import json,math
from typing import List,Optional,Tuple
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import Image,CameraInfo,LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker,MarkerArray

class FireFusionNode(Node):
    def __init__(self):
        super().__init__('fire_fusion_node')
        self.bridge=CvBridge()

        self.declare_parameter('target_frame','base_link')
        self.target_frame=self.get_parameter('target_frame').value

        self.declare_parameter('map_frame','map')
        self.map_frame=self.get_parameter('map_frame').value

        self.declare_parameter('lidar_depth_tolerance',0.5)
        self.lidar_depth_tolerance=self.get_parameter('lidar_depth_tolerance').value

        self.declare_parameter('min_depth',0.2)
        self.min_depth=self.get_parameter('min_depth').value

        self.declare_parameter('max_depth',8.0)
        self.max_depth=self.get_parameter('max_depth').value

        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.fx=600.0
        self.fy=600.0
        self.cx=320.0
        self.cy=240.0
        self.camera_info_received=False
        self.camera_frame_id='camera_color_optical_frame'

        self.latest_depth_img:Optional[np.ndarray]=None
        self.latest_depth_encoding='16UC1'
        self.latest_depth_stamp=None
        self.latest_scan:Optional[LaserScan]=None

        sensor_qos=QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub_candidates=self.create_subscription(
            String,
            '/fire_candidates',
            self._candidates_callback,
            10
        )

        self.sub_depth=self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self._depth_callback,
            sensor_qos
        )

        self.sub_info=self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self._camera_info_callback,
            10
        )

        self.sub_scan=self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            sensor_qos
        )

        self.pub_fire_tracks=self.create_publisher(
            String,
            '/fire_tracks_3d',
            10
        )

        self.pub_alarm=self.create_publisher(
            Bool,
            '/fire_alarm',
            10
        )

        self.pub_markers=self.create_publisher(
            MarkerArray,
            '/fire_fusion/debug_markers',
            10
        )

        self.get_logger().info(
            f'Fire Fusion Node started: '
            f'{self.camera_frame_id} -> '
            f'{self.target_frame} -> '
            f'{self.map_frame}'
        )

    def _camera_info_callback(self,msg:CameraInfo):
        self.fx=msg.k[0]
        self.fy=msg.k[4]
        self.cx=msg.k[2]
        self.cy=msg.k[5]

        if msg.header.frame_id:
            self.camera_frame_id=msg.header.frame_id

        if not self.camera_info_received:
            self.camera_info_received=True
            self.get_logger().info(
                f'CameraInfo loaded: '
                f'fx={self.fx:.1f}, '
                f'fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, '
                f'cy={self.cy:.1f}, '
                f'frame={self.camera_frame_id}'
            )

    def _depth_callback(self,msg:Image):
        try:
            if msg.encoding in ['16UC1','mono16']:
                self.latest_depth_img=self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='passthrough'
                )
                self.latest_depth_encoding='16UC1'

            elif msg.encoding=='32FC1':
                self.latest_depth_img=self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='32FC1'
                )
                self.latest_depth_encoding='32FC1'

            self.latest_depth_stamp=msg.header.stamp

        except Exception as e:
            self.get_logger().error(
                f'Depth conversion error: {e}'
            )

    def _scan_callback(self,msg:LaserScan):
        self.latest_scan=msg

    def _get_tf(self,target_frame,source_frame,stamp):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp
            )
        except(
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time()
                )
            except(
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException
            ) as e:
                self.get_logger().debug(
                    f'TF unavailable: '
                    f'{source_frame} -> {target_frame}: {e}'
                )
                return None

    def _candidates_callback(self,msg:String):
        if self.latest_depth_img is None:
            return

        try:
            payload=json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(
                f'Fire candidate JSON error: {e}'
            )
            return

        header=payload.get('header',{})
        stamp_value=header.get('stamp',0.0)
        candidates=payload.get('candidates',[])

        if not candidates:
            self._publish_alarm(False)
            self._publish_tracks(
                stamp_value,
                []
            )
            self._publish_markers([])
            return

        stamp=self._float_to_ros_time(stamp_value)

        tf_camera_to_base=self._get_tf(
            self.target_frame,
            self.camera_frame_id,
            stamp
        )

        if tf_camera_to_base is None:
            self.get_logger().warn(
                f'Camera TF unavailable: '
                f'{self.camera_frame_id} -> '
                f'{self.target_frame}'
            )
            return

        fire_tracks=[]

        for candidate in candidates:
            try:
                candidate_id=int(
                    candidate['candidate_id']
                )
                bbox=candidate['bbox']
                confidence=float(
                    candidate.get('confidence',0.0)
                )
            except(
                KeyError,
                TypeError,
                ValueError
            ):
                continue

            result=self._calculate_3d_from_depth(
                bbox
            )

            if result is None:
                continue

            cam_x,cam_y,cam_z=result

            lidar_distance=self._get_lidar_distance(
                cam_x,
                cam_z
            )

            fusion_distance=cam_z
            lidar_valid=False

            if lidar_distance is not None:
                if abs(
                    lidar_distance-cam_z
                )<=self.lidar_depth_tolerance:
                    fusion_distance=(
                        0.7*cam_z+
                        0.3*lidar_distance
                    )
                    lidar_valid=True

            scale=fusion_distance/cam_z
            cam_x*=scale
            cam_y*=scale
            cam_z=fusion_distance

            plane_valid=self._check_plane_geometry(
                bbox,
                cam_z
            )

            if(
                self.latest_scan is not None
                and lidar_distance is not None
                and not lidar_valid
            ):
                continue

            if not plane_valid:
                continue

            p_cam=PointStamped()
            p_cam.header.frame_id=self.camera_frame_id
            p_cam.header.stamp=stamp
            p_cam.point.x=cam_x
            p_cam.point.y=cam_y
            p_cam.point.z=cam_z

            try:
                p_base=do_transform_point(
                    p_cam,
                    tf_camera_to_base
                )
            except Exception as e:
                self.get_logger().warn(
                    f'Camera -> base transform failed: {e}'
                )
                continue

            base_x=p_base.point.x
            base_y=p_base.point.y
            base_z=p_base.point.z

            map_position=None

            tf_base_to_map=self._get_tf(
                self.map_frame,
                self.target_frame,
                stamp
            )

            if tf_base_to_map is not None:
                try:
                    p_map=do_transform_point(
                        p_base,
                        tf_base_to_map
                    )

                    map_position=[
                        round(float(p_map.point.x),2),
                        round(float(p_map.point.y),2),
                        round(float(p_map.point.z),2)
                    ]

                except Exception as e:
                    self.get_logger().debug(
                        f'Base -> map transform failed: {e}'
                    )

            fire_track={
                'fire_id':candidate_id,
                'bbox':bbox,

                # base_link 기준
                'position':[
                    round(float(base_x),2),
                    round(float(base_y),2),
                    round(float(base_z),2)
                ],

                # map 기준
                'position_map':map_position,

                'distance':round(
                    float(
                        math.hypot(
                            base_x,
                            base_y
                        )
                    ),
                    2
                ),

                'confidence':round(
                    confidence,
                    3
                ),

                'depth':round(
                    float(cam_z),
                    2
                ),

                'lidar_distance':(
                    round(
                        float(lidar_distance),
                        2
                    )
                    if lidar_distance is not None
                    else None
                ),

                'lidar_valid':lidar_valid,
                'plane_valid':plane_valid,
                'temporal_hits':candidate.get(
                    'temporal_hits',
                    0
                ),
                'stamp':stamp_value
            }

            fire_tracks.append(
                fire_track
            )

        self._publish_tracks(
            stamp_value,
            fire_tracks
        )

        self._publish_alarm(
            len(fire_tracks)>0
        )

        self._publish_markers(
            fire_tracks
        )

        if fire_tracks:
            for fire in fire_tracks:
                self.get_logger().warn(
                    f'FIRE DETECTED '
                    f'id={fire["fire_id"]} '
                    f'confidence={fire["confidence"]:.2f} '
                    f'base={fire["position"]} '
                    f'map={fire["position_map"]}'
                )

    def _float_to_ros_time(self,value):
        try:
            sec=int(float(value))
            nanosec=int(
                (float(value)-sec)*1e9
            )

            return rclpy.time.Time(
                seconds=sec,
                nanoseconds=nanosec
            )
        except Exception:
            return rclpy.time.Time()

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
        msg.data=json.dumps(
            output,
            ensure_ascii=False
        )

        self.pub_fire_tracks.publish(
            msg
        )

    def _publish_alarm(self,state):
        msg=Bool()
        msg.data=bool(state)
        self.pub_alarm.publish(msg)

    def _calculate_3d_from_depth(
        self,
        bbox:List[float]
    )->Optional[Tuple[float,float,float]]:

        h_img,w_img=(
            self.latest_depth_img.shape[:2]
        )

        xmin,ymin,xmax,ymax=map(
            int,
            bbox
        )

        xmin=max(
            0,
            min(w_img-1,xmin)
        )

        xmax=max(
            0,
            min(w_img,xmax)
        )

        ymin=max(
            0,
            min(h_img-1,ymin)
        )

        ymax=max(
            0,
            min(h_img,ymax)
        )

        if xmax<=xmin or ymax<=ymin:
            return None

        cx_box=(
            xmin+xmax
        )/2.0

        cy_box=(
            ymin+ymax
        )/2.0

        w_box=(
            xmax-xmin
        )*0.5

        h_box=(
            ymax-ymin
        )*0.5

        rx1=max(
            0,
            int(cx_box-w_box/2)
        )

        rx2=min(
            w_img,
            int(cx_box+w_box/2)
        )

        ry1=max(
            0,
            int(cy_box-h_box/2)
        )

        ry2=min(
            h_img,
            int(cy_box+h_box/2)
        )

        if rx2<=rx1 or ry2<=ry1:
            return None

        depth_roi=self.latest_depth_img[
            ry1:ry2,
            rx1:rx2
        ]

        if self.latest_depth_encoding=='16UC1':
            valid_depths=(
                depth_roi[
                    depth_roi>0
                ]/1000.0
            )
        else:
            valid_depths=depth_roi[
                np.isfinite(depth_roi)
                &(depth_roi>0.1)
            ]

        if len(valid_depths)<10:
            return None

        z_cam=float(
            np.median(valid_depths)
        )

        if(
            z_cam<self.min_depth
            or z_cam>self.max_depth
        ):
            return None

        x_cam=(
            (cx_box-self.cx)
            *z_cam
            /self.fx
        )

        y_cam=(
            (cy_box-self.cy)
            *z_cam
            /self.fy
        )

        return x_cam,y_cam,z_cam

    def _get_lidar_distance(
        self,
        cam_x:float,
        cam_z:float
    )->Optional[float]:

        if self.latest_scan is None:
            return None

        angle_rad=-math.atan2(
            cam_x,
            cam_z
        )

        scan=self.latest_scan

        if(
            angle_rad<scan.angle_min
            or angle_rad>scan.angle_max
        ):
            return None

        idx=int(
            (angle_rad-scan.angle_min)
            /scan.angle_increment
        )

        if not(
            0<=idx<len(scan.ranges)
        ):
            return None

        scan_dist=scan.ranges[idx]

        if not(
            scan.range_min
            <=scan_dist
            <=scan.range_max
        ):
            return None

        return float(scan_dist)

    def _check_plane_geometry(
        self,
        bbox:List[float],
        representative_depth:float
    )->bool:

        h_img,w_img=(
            self.latest_depth_img.shape[:2]
        )

        xmin,ymin,xmax,ymax=map(
            int,
            bbox
        )

        xmin=max(0,xmin)
        ymin=max(0,ymin)
        xmax=min(w_img,xmax)
        ymax=min(h_img,ymax)

        if xmax<=xmin or ymax<=ymin:
            return False

        roi=self.latest_depth_img[
            ymin:ymax,
            xmin:xmax
        ]

        if roi.size==0:
            return False

        step_y=max(
            1,
            roi.shape[0]//15
        )

        step_x=max(
            1,
            roi.shape[1]//15
        )

        ys,xs=np.mgrid[
            ymin:ymax:step_y,
            xmin:xmax:step_x
        ]

        xs=xs.flatten()
        ys=ys.flatten()

        if len(xs)<10:
            return True

        depth_values=[]

        for px,py in zip(xs,ys):
            depth=self.latest_depth_img[
                py,
                px
            ]

            if self.latest_depth_encoding=='16UC1':
                depth=float(depth)/1000.0
            else:
                depth=float(depth)

            if(
                not np.isfinite(depth)
                or depth<self.min_depth
                or depth>self.max_depth
            ):
                continue

            depth_values.append(
                (
                    float(px),
                    float(py),
                    depth
                )
            )

        if len(depth_values)<10:
            return True

        points=[]

        for px,py,z in depth_values:
            x=(
                (px-self.cx)
                *z
                /self.fx
            )

            y=(
                (py-self.cy)
                *z
                /self.fy
            )

            points.append(
                [x,y,z]
            )

        points=np.asarray(
            points,
            dtype=np.float64
        )

        center=np.mean(
            points,
            axis=0
        )

        centered=points-center

        try:
            _,_,vh=np.linalg.svd(
                centered
            )
        except np.linalg.LinAlgError:
            return True

        normal=vh[-1]

        residuals=np.abs(
            centered@normal
        )

        median_residual=float(
            np.median(residuals)
        )

        p95_residual=float(
            np.percentile(
                residuals,
                95
            )
        )

        plane_like=(
            median_residual<0.015
            and p95_residual<0.05
        )

        return not plane_like

    def _publish_markers(self,fires):
        marker_array=MarkerArray()

        for fire in fires:
            marker=Marker()

            marker.header.frame_id=self.target_frame
            marker.header.stamp=(
                self.get_clock().now().to_msg()
            )

            marker.ns='fire_tracks'
            marker.id=int(
                fire['fire_id']
            )

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

            marker_array.markers.append(
                marker
            )

        self.pub_markers.publish(
            marker_array
        )

def main(args=None):
    rclpy.init(args=args)
    node=FireFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Fire Fusion Node stopped.'
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()