#!/usr/bin/env python3
"""
Node1: FireDetectionNode

SUB:
    /camera/image_raw
    /camera/depth/image_raw

PUB:
    /fire_candidates
    /camera/fire_detection/image
"""

import json,math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')
        self.bridge=CvBridge()
        self.window_open=False

        self.declare_parameter('confirm_frames',10)
        self.declare_parameter('max_missed_frames',1)
        self.declare_parameter('bbox_iou_threshold',0.25)
        self.declare_parameter('center_distance_threshold',80.0)

        self.confirm_frames=int(self.get_parameter('confirm_frames').value)
        self.max_missed_frames=int(self.get_parameter('max_missed_frames').value)
        self.bbox_iou_threshold=float(self.get_parameter('bbox_iou_threshold').value)
        self.center_distance_threshold=float(self.get_parameter('center_distance_threshold').value)

        self.temporal_tracks={}
        self.next_track_id=0

        self.rgb_sub=message_filters.Subscriber(self,Image,'/camera/image_raw')
        self.depth_sub=message_filters.Subscriber(self,Image,'/camera/depth/image_raw')

        self.ts=message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub,self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        self.candidate_pub=self.create_publisher(
            String,'/fire_candidates',10
        )

        self.image_pub=self.create_publisher(
            Image,'/camera/fire_detection/image',10
        )

        self.get_logger().info(
            f'Fire Detection Node started. '
            f'confirmation={self.confirm_frames} frames'
        )

    @staticmethod
    def bbox_iou(a,b):
        ax1,ay1,ax2,ay2=a
        bx1,by1,bx2,by2=b

        ix1=max(ax1,bx1)
        iy1=max(ay1,by1)
        ix2=min(ax2,bx2)
        iy2=min(ay2,by2)

        inter=max(0,ix2-ix1)*max(0,iy2-iy1)

        area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
        area_b=max(0,bx2-bx1)*max(0,by2-by1)

        union=area_a+area_b-inter

        if union<=0:
            return 0.0

        return float(inter/union)

    @staticmethod
    def bbox_center(bbox):
        return(
            (bbox[0]+bbox[2])/2.0,
            (bbox[1]+bbox[3])/2.0
        )

    @classmethod
    def center_distance(cls,a,b):
        ax,ay=cls.bbox_center(a)
        bx,by=cls.bbox_center(b)
        return math.hypot(ax-bx,ay-by)

    def _match_candidate(self,candidate,matched_ids):
        best_id=None
        best_score=-1.0

        bbox=candidate['bbox']

        for track_id,track in self.temporal_tracks.items():
            if track_id in matched_ids:
                continue

            old_bbox=track['bbox']

            iou=self.bbox_iou(bbox,old_bbox)
            distance=self.center_distance(bbox,old_bbox)

            if(
                iou<self.bbox_iou_threshold
                and distance>self.center_distance_threshold
            ):
                continue

            distance_score=max(
                0.0,
                1.0-distance/self.center_distance_threshold
            )

            score=0.7*iou+0.3*distance_score

            if score>best_score:
                best_score=score
                best_id=track_id

        return best_id

    def _update_temporal_tracks(self,candidates):
        matched_ids=set()
        confirmed_candidates=[]

        for candidate in candidates:
            track_id=self._match_candidate(
                candidate,
                matched_ids
            )

            if track_id is not None:
                track=self.temporal_tracks[track_id]

                track['bbox']=candidate['bbox']
                track['confidence']=candidate['confidence']
                track['area']=candidate['area']
                track['aspect_ratio']=candidate['aspect_ratio']

                track['hit_count']+=1
                track['missed_count']=0

                matched_ids.add(track_id)

                if track['hit_count']>=self.confirm_frames:
                    confirmed=dict(candidate)
                    confirmed['candidate_id']=track_id
                    confirmed['temporal_hits']=track['hit_count']

                    confirmed_candidates.append(
                        confirmed
                    )

            else:
                track_id=self.next_track_id
                self.next_track_id+=1

                self.temporal_tracks[track_id]={
                    'bbox':candidate['bbox'],
                    'confidence':candidate['confidence'],
                    'area':candidate['area'],
                    'aspect_ratio':candidate['aspect_ratio'],
                    'hit_count':1,
                    'missed_count':0
                }

                matched_ids.add(track_id)

        delete_ids=[]

        for track_id,track in self.temporal_tracks.items():
            if track_id in matched_ids:
                continue

            track['missed_count']+=1

            if(
                track['missed_count']
                >self.max_missed_frames
            ):
                delete_ids.append(track_id)

        for track_id in delete_ids:
            del self.temporal_tracks[track_id]

        return confirmed_candidates

    def image_callback(self,rgb_msg,depth_msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(
                f'RGB conversion failed: {e}'
            )
            return

        hsv=cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2HSV
        )

        lower_bound1=np.array([0,70,70])
        upper_bound1=np.array([10,255,255])

        lower_bound2=np.array([170,120,120])
        upper_bound2=np.array([179,255,255])

        mask1=cv2.inRange(
            hsv,
            lower_bound1,
            upper_bound1
        )

        mask2=cv2.inRange(
            hsv,
            lower_bound2,
            upper_bound2
        )

        mask=cv2.bitwise_or(mask1,mask2)

        kernel=cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5,5)
        )

        mask=cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask=cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours,_=cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates=[]
        min_area=500

        for contour in contours:
            area=cv2.contourArea(contour)

            if area<min_area:
                continue

            x,y,w,h=cv2.boundingRect(contour)

            if w<=0 or h<=0:
                continue

            aspect_ratio=h/float(w)

            if aspect_ratio<0.25:
                continue

            bbox_area=w*h

            fill_ratio=area/float(bbox_area)
            fill_ratio=max(
                0.0,
                min(1.0,fill_ratio)
            )

            area_score=min(
                area/5000.0,
                1.0
            )

            confidence=(
                0.6*fill_ratio+
                0.4*area_score
            )

            candidates.append({
                'bbox':[
                    int(x),
                    int(y),
                    int(x+w),
                    int(y+h)
                ],
                'confidence':round(
                    float(confidence),
                    3
                ),
                'area':round(
                    float(area),
                    1
                ),
                'aspect_ratio':round(
                    float(aspect_ratio),
                    3
                )
            })

        confirmed_candidates=self._update_temporal_tracks(
            candidates
        )

        stamp=(
            rgb_msg.header.stamp.sec+
            rgb_msg.header.stamp.nanosec*1e-9
        )

        candidate_msg=String()

        candidate_msg.data=json.dumps(
            {
                'header':{
                    'stamp':stamp,
                    'frame_id':rgb_msg.header.frame_id
                },
                'candidates':confirmed_candidates
            },
            ensure_ascii=False
        )

        self.candidate_pub.publish(
            candidate_msg
        )

        # 5프레임 연속 확인된 후보만 표시
        for candidate in confirmed_candidates:
            x1,y1,x2,y2=map(
                int,
                candidate['bbox']
            )

            track_id=candidate['candidate_id']
            confidence=candidate['confidence']
            hits=candidate['temporal_hits']

            cv2.rectangle(
                cv_image,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                3
            )

            cv2.putText(
                cv_image,
                f'FIRE CANDIDATE ID:{track_id} '
                f'{confidence:.2f} '
                f'{hits}F',
                (x1,max(y1-10,25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

        # Debug image publish
        try:
            processed_msg=self.bridge.cv2_to_imgmsg(
                cv_image,
                encoding='bgr8'
            )

            processed_msg.header=rgb_msg.header

            self.image_pub.publish(
                processed_msg
            )

        except Exception as e:
            self.get_logger().error(
                f'Failed to publish image: {e}'
            )

def main(args=None):
    rclpy.init(args=args)
    node=FireDetectionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            'Fire Detection Node stopped.'
        )

    except Exception as e:
        node.get_logger().error(
            f'Exception: {e}'
        )

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()