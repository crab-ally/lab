#!/usr/bin/env python3
"""
Fall Detection YOLO Pose 데이터셋 생성기

Classes:
0: fallen
1: standing
2: bending
3: sitting

Keypoints: 12
0: head
1: left_shoulder
2: right_shoulder
3: left_elbow
4: right_elbow
5: left_wrist
6: right_wrist
7: pelvis
8: left_knee
9: right_knee
10: left_ankle
11: right_ankle

출력:
fall_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── debug/
└── metadata.json

YOLO Pose label:
class cx cy w h
kp0_x kp0_y kp0_v
kp1_x kp1_y kp1_v
...
kp11_x kp11_y kp11_v
"""

import mujoco
import numpy as np
import cv2
import os
import json
import random

# ============================================================
# Path
# ============================================================

XML_PATH="/workspace/worlds/dataset/fall_dataset_world.xml"
DATASET="/workspace/datasets/fall_dataset"

TRAIN_IMAGE=os.path.join(DATASET,"images/train")
VAL_IMAGE=os.path.join(DATASET,"images/val")
TRAIN_LABEL=os.path.join(DATASET,"labels/train")
VAL_LABEL=os.path.join(DATASET,"labels/val")
META_PATH=os.path.join(DATASET,"metadata.json")

# ============================================================
# Debug
# ============================================================

DEBUG_POSE=False
DEBUG_IMAGE_DIR=os.path.join(DATASET,"debug")

for path in [TRAIN_IMAGE,VAL_IMAGE,TRAIN_LABEL,VAL_LABEL]:
    os.makedirs(path,exist_ok=True)

if DEBUG_POSE:
    os.makedirs(DEBUG_IMAGE_DIR,exist_ok=True)

CLASS_NAMES={
    0:"fallen",
    1:"standing",
    2:"bending",
    3:"sitting",
}

# ============================================================
# YOLO Pose
# ============================================================

NUM_KEYPOINTS=12

KEYPOINT_NAMES=[
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "pelvis",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# ============================================================
# Load MuJoCo
# ============================================================

model=mujoco.MjModel.from_xml_path(XML_PATH)
data=mujoco.MjData(model)

cam_id=mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "dataset_camera"
)

if cam_id<0:
    raise RuntimeError("dataset_camera not found")

# ============================================================
# Workers
# ============================================================

workers_info=[]

categories=[
    ("fallen_faceup",2,0,"ffu",True),
    ("fallen_facedown",2,0,"ffd",True),
    ("standing_worker",2,1,"sw",True),
    ("bending_worker",2,2,"bw",True),
    ("sitting_worker",2,3,"stw",False),
]

for name_prefix,count,cls_id,geom_prefix,rand_pos in categories:
    for i in range(count):
        body_name=f"{name_prefix}_{i}"

        bid=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )

        if bid<0:
            raise RuntimeError(f"{body_name} not found")

        initial_pos=np.array(model.body_pos[bid],dtype=np.float64)
        initial_quat=np.array(model.body_quat[bid],dtype=np.float64)

        workers_info.append({
            "name":body_name,
            "id":bid,
            "class_id":cls_id,
            "geom_prefix":geom_prefix,
            "rand_pos":rand_pos,
            "initial_pos":initial_pos,
            "initial_quat":initial_quat,
            "worker_idx":i
        })

# ============================================================
# Renderer
# ============================================================

WIDTH=640
HEIGHT=480

renderer=mujoco.Renderer(
    model,
    HEIGHT,
    WIDTH
)

# ============================================================
# Pose visibility
# ============================================================

MIN_VISIBLE_KEYPOINTS=6

# keypoint별 visibility 판정 거리 여유
RAY_EPS=0.05

# ============================================================
# Random color
# ============================================================

def random_color(base):
    return [
        np.clip(
            c+np.random.uniform(-0.15,0.15),
            0,
            1
        )
        for c in base
    ]

# ============================================================
# World -> Pixel
# ============================================================

def project(point):
    cam_pos=data.cam_xpos[cam_id]
    R=data.cam_xmat[cam_id].reshape(3,3)

    pc=R.T@(np.asarray(point)-cam_pos)

    x,y,z=pc

    depth=-z

    if depth<=1e-6:
        return None

    fovy=model.cam_fovy[cam_id]

    f=HEIGHT/(2*np.tan(np.deg2rad(fovy)/2))

    u=WIDTH/2+f*x/depth
    v=HEIGHT/2-f*y/depth

    return float(u),float(v)

# ============================================================
# Collision avoidance
# ============================================================

def is_valid_position(x,y,radius=0.5):

    for i in range(model.ngeom):

        name=mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            i
        )

        if name:
            if any(
                name.startswith(p)
                for p in ["ffu","ffd","sw","bw","stw"]
            ):
                continue

            if name=="floor":
                continue

        pos=data.geom_xpos[i]

        size=model.geom_size[i]
        gtype=model.geom_type[i]

        dx=abs(x-pos[0])
        dy=abs(y-pos[1])

        if gtype==mujoco.mjtGeom.mjGEOM_BOX:

            if (
                dx<size[0]+radius
                and
                dy<size[1]+radius
            ):
                return False

        elif gtype in (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_SPHERE
        ):

            if np.hypot(dx,dy)<size[0]+radius:
                return False

    return True

# ============================================================
# Geometry ID
# ============================================================

def get_geom(prefix,part):

    name=f"{prefix}{part}"

    gid=mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name
    )

    if gid<0:
        return None

    return gid

# ============================================================
# Capsule endpoints
# ============================================================

def capsule_endpoints(gid):

    center=data.geom_xpos[gid]

    R=data.geom_xmat[gid].reshape(3,3)

    half_length=model.geom_size[gid][1]

    p1=center-R[:,2]*half_length
    p2=center+R[:,2]*half_length

    return p1,p2

# ============================================================
# Capsule center
# ============================================================

def geom_center(gid):
    return np.array(
        data.geom_xpos[gid],
        dtype=np.float64
    )

# ============================================================
# Get pose keypoints in world coordinates
# ============================================================

def get_worker_keypoints(prefix):

    kp=[None]*NUM_KEYPOINTS

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    gid=get_geom(prefix,"head")

    if gid is not None:
        kp[0]=geom_center(gid)

    # --------------------------------------------------------
    # PELVIS
    # --------------------------------------------------------

    gid=get_geom(prefix,"pelvis")

    if gid is not None:
        p1,p2=capsule_endpoints(gid)
        kp[7]=(p1+p2)/2

    # --------------------------------------------------------
    # SHOULDERS / ELBOWS / WRISTS
    #
    # arm capsule:
    # shoulder -> wrist
    #
    # elbow = 중간점
    # --------------------------------------------------------

    for side,shoulder_idx,elbow_idx,wrist_idx in [
        ("larm",1,3,5),
        ("rarm",2,4,6)
    ]:

        gid=get_geom(prefix,side)

        if gid is None:
            continue

        p1,p2=capsule_endpoints(gid)

        # XML의 fromto 방향과 실제 geom axis 방향이
        # 항상 동일하다는 보장이 없으므로
        # 두 endpoint 중 torso에 가까운 쪽을 shoulder로 사용.
        torso_gid=get_geom(prefix,"torso")

        if torso_gid is not None:

            torso_center=geom_center(torso_gid)

            if np.linalg.norm(p1-torso_center)<np.linalg.norm(p2-torso_center):
                shoulder=p1
                wrist=p2
            else:
                shoulder=p2
                wrist=p1

        else:
            shoulder=p1
            wrist=p2

        elbow=(shoulder+wrist)/2

        kp[shoulder_idx]=shoulder
        kp[elbow_idx]=elbow
        kp[wrist_idx]=wrist

    # --------------------------------------------------------
    # KNEES / ANKLES
    #
    # upper leg:
    # hip -> knee
    #
    # lower leg:
    # knee -> ankle
    # --------------------------------------------------------

    for side,upper_name,lower_name,knee_idx,ankle_idx in [
        (
            "left",
            "luleg",
            "llleg",
            8,
            10
        ),
        (
            "right",
            "ruleg",
            "rlleg",
            9,
            11
        )
    ]:

        upper_gid=get_geom(prefix,upper_name)
        lower_gid=get_geom(prefix,lower_name)

        if upper_gid is None or lower_gid is None:
            continue

        upper_p1,upper_p2=capsule_endpoints(
            upper_gid
        )

        lower_p1,lower_p2=capsule_endpoints(
            lower_gid
        )

        # 두 capsule에서 가장 가까운 endpoint를 knee로 찾는다.
        candidates=[
            (upper_p1,lower_p1),
            (upper_p1,lower_p2),
            (upper_p2,lower_p1),
            (upper_p2,lower_p2)
        ]

        best=min(
            candidates,
            key=lambda x:np.linalg.norm(x[0]-x[1])
        )

        knee=(best[0]+best[1])/2

        # lower leg의 knee 반대쪽 endpoint가 ankle
        d1=np.linalg.norm(lower_p1-knee)
        d2=np.linalg.norm(lower_p2-knee)

        if d1>d2:
            ankle=lower_p1
        else:
            ankle=lower_p2

        kp[knee_idx]=knee
        kp[ankle_idx]=ankle

    return kp

# ============================================================
# Keypoint visibility
# ============================================================

def is_keypoint_visible(point,owner_prefix):

    if point is None:
        return False

    pix=project(point)

    if pix is None:
        return False

    u,v=pix

    if not (
        0<=u<WIDTH
        and
        0<=v<HEIGHT
    ):
        return False

    cam_pos=data.cam_xpos[cam_id]

    vec=np.asarray(point)-cam_pos

    dist=np.linalg.norm(vec)

    if dist<1e-6:
        return True

    vec_norm=vec/dist

    geomid=np.array(
        [-1],
        dtype=np.int32
    )

    hit_dist=mujoco.mj_ray(
        model,
        data,
        cam_pos,
        vec_norm,
        None,
        1,
        -1,
        geomid
    )

    if hit_dist<0:
        return True

    if hit_dist>=dist-RAY_EPS:
        return True

    if geomid[0]!=-1:

        hit_name=mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geomid[0]
        )

        if (
            hit_name
            and
            hit_name.startswith(owner_prefix)
        ):
            return True

    return False

# ============================================================
# Pose projection + visibility
# ============================================================

def calculate_pose(prefix):

    world_keypoints=get_worker_keypoints(prefix)

    image_keypoints=[]

    for point in world_keypoints:

        if point is None:

            image_keypoints.append(
                (0.0,0.0,0)
            )

            continue

        pix=project(point)

        if pix is None:

            image_keypoints.append(
                (0.0,0.0,0)
            )

            continue

        u,v=pix

        visible=is_keypoint_visible(
            point,
            prefix
        )

        if not visible:

            image_keypoints.append(
                (
                    float(u),
                    float(v),
                    0
                )
            )

        else:

            image_keypoints.append(
                (
                    float(u),
                    float(v),
                    2
                )
            )

    return world_keypoints,image_keypoints

# ============================================================
# Bounding box from keypoints
# ============================================================

def bbox_from_keypoints(keypoints):

    visible=[
        (x,y)
        for x,y,v in keypoints
        if v>0
    ]

    if len(visible)<2:
        return None

    xs=[p[0] for p in visible]
    ys=[p[1] for p in visible]

    x1=np.clip(
        min(xs),
        0,
        WIDTH-1
    )

    y1=np.clip(
        min(ys),
        0,
        HEIGHT-1
    )

    x2=np.clip(
        max(xs),
        0,
        WIDTH-1
    )

    y2=np.clip(
        max(ys),
        0,
        HEIGHT-1
    )

    # 실제 사람 geometry보다 keypoint만으로 bbox를
    # 만들면 약간 작아질 수 있으므로 padding 추가.
    pad_x=(x2-x1)*0.10
    pad_y=(y2-y1)*0.10

    x1=np.clip(
        x1-pad_x,
        0,
        WIDTH-1
    )

    y1=np.clip(
        y1-pad_y,
        0,
        HEIGHT-1
    )

    x2=np.clip(
        x2+pad_x,
        0,
        WIDTH-1
    )

    y2=np.clip(
        y2+pad_y,
        0,
        HEIGHT-1
    )

    if x2<=x1 or y2<=y1:
        return None

    if x2-x1<6 or y2-y1<6:
        return None

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )

# ============================================================
# Random Scene
# ============================================================

def randomize_scene():

    active_workers=[]

    fallen_candidates=[
        w for w in workers_info
        if w["class_id"]==0
    ]

    other_candidates=[
        w for w in workers_info
        if w["class_id"]!=0
    ]

    n_fallen=random.randint(
        1,
        len(fallen_candidates)
    )

    active_fallen=random.sample(
        fallen_candidates,
        n_fallen
    )

    n_other=random.randint(
        0,
        len(other_candidates)
    )

    active_other=random.sample(
        other_candidates,
        n_other
    )

    selected_workers=active_fallen+active_other

    placed_positions=[]

    for w in workers_info:

        bid=w["id"]

        if w not in selected_workers:

            model.body_pos[bid]=[
                100.0,
                100.0,
                -10.0
            ]

            continue

        w_x,w_y=0.0,0.0

        if w["rand_pos"]:

            found=False

            for _ in range(100):

                x=np.random.uniform(
                    -4.0,
                    4.0
                )

                y=np.random.uniform(
                    -4.0,
                    4.0
                )

                if not is_valid_position(
                    x,
                    y,
                    0.6
                ):
                    continue

                overlap=False

                for px,py in placed_positions:

                    if np.hypot(
                        x-px,
                        y-py
                    )<1.0:

                        overlap=True
                        break

                if overlap:
                    continue

                w_x=x
                w_y=y
                found=True
                break

            if not found:
                continue

            model.body_pos[bid]=[
                w_x,
                w_y,
                w["initial_pos"][2]
            ]

            yaw=np.random.uniform(
                0,
                2*np.pi
            )

            yaw_quat=np.zeros(4)

            mujoco.mju_axisAngle2Quat(
                yaw_quat,
                np.array([
                    0.0,
                    0.0,
                    1.0
                ]),
                yaw
            )

            result_quat=np.zeros(4)

            mujoco.mju_mulQuat(
                result_quat,
                yaw_quat,
                w["initial_quat"]
            )

            model.body_quat[bid]=result_quat

        else:

            model.body_pos[bid]=w["initial_pos"]
            model.body_quat[bid]=w["initial_quat"]

            w_x=w["initial_pos"][0]
            w_y=w["initial_pos"][1]

        placed_positions.append(
            (w_x,w_y)
        )

        # ----------------------------------------------------
        # Random clothing color
        # ----------------------------------------------------

        color=random_color(
            np.array([
                0.2,
                0.24,
                0.28
            ])
        )

        geom_prefix_full=(
            f"{w['geom_prefix']}"
            f"{w['worker_idx']}_"
        )

        for part in [
            "torso",
            "pelvis",
            "luleg",
            "llleg",
            "ruleg",
            "rlleg",
            "larm",
            "rarm"
        ]:

            gid=get_geom(
                geom_prefix_full,
                part
            )

            if gid is not None:

                model.geom_rgba[gid][:3]=color

        active_workers.append({
            "name":w["name"],
            "class_id":w["class_id"],
            "geom_prefix":geom_prefix_full,
            "x":float(w_x),
            "y":float(w_y)
        })

    # ========================================================
    # Camera
    # ========================================================

    CAM_Z=0.383

    fallen_workers=[
        w for w in active_workers
        if w["class_id"]==0
    ]

    if not fallen_workers:
        raise RuntimeError(
            "No fallen worker exists"
        )

    target=random.choice(
        fallen_workers
    )

    cam_x=target["x"]
    cam_y=target["y"]

    for _ in range(100):

        candidate_x=np.random.uniform(
            -4.0,
            4.0
        )

        candidate_y=np.random.uniform(
            -4.0,
            4.0
        )

        if not is_valid_position(
            candidate_x,
            candidate_y,
            0.3
        ):
            continue

        dist=np.hypot(
            target["x"]-candidate_x,
            target["y"]-candidate_y
        )

        if 2.0<=dist<=7.0:

            cam_x=candidate_x
            cam_y=candidate_y

            break

    model.cam_pos[cam_id]=[
        cam_x,
        cam_y,
        CAM_Z
    ]

    # ========================================================
    # Camera direction
    # ========================================================

    dx=target["x"]-cam_x
    dy=target["y"]-cam_y

    yaw=np.arctan2(
        dy,
        dx
    )

    yaw+=np.random.uniform(
        np.deg2rad(-20),
        np.deg2rad(20)
    )

    base_quat=np.array([
        0.5,
        0.5,
        -0.5,
        -0.5
    ])

    yaw_quat=np.zeros(4)

    mujoco.mju_axisAngle2Quat(
        yaw_quat,
        np.array([0,0,1]),
        yaw
    )

    result=np.zeros(4)

    mujoco.mju_mulQuat(
        result,
        yaw_quat,
        base_quat
    )

    model.cam_quat[cam_id]=result

    return active_workers

# ============================================================
# Debug drawing
# ============================================================

def draw_pose(img,keypoints,class_id):

    # skeleton
    skeleton=[
        (0,1),
        (0,2),
        (1,3),
        (3,5),
        (2,4),
        (4,6),
        (1,7),
        (2,7),
        (7,8),
        (8,10),
        (7,9),
        (9,11),
    ]

    color_map=[
        (0,0,255),
        (0,220,50),
        (0,255,255),
        (255,0,255)
    ]

    color=color_map[
        class_id%len(color_map)
    ]

    for a,b in skeleton:

        xa,ya,va=keypoints[a]
        xb,yb,vb=keypoints[b]

        if va>0 and vb>0:

            cv2.line(
                img,
                (int(xa),int(ya)),
                (int(xb),int(yb)),
                color,
                2
            )

    for idx,(x,y,v) in enumerate(keypoints):

        if v<=0:
            continue

        cv2.circle(
            img,
            (int(x),int(y)),
            4,
            color,
            -1
        )

        cv2.putText(
            img,
            str(idx),
            (int(x)+4,int(y)-4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

# ============================================================
# Dataset generation
# ============================================================

NUM_DATA=10000
TRAIN_RATIO=0.8

CLASS_FALLEN=0
CLASS_STANDING=1
CLASS_BENDING=2
CLASS_SITTING=3

indices=list(range(NUM_DATA))

random.seed(42)
np.random.seed(42)

random.shuffle(indices)

train_set=set(
    indices[:int(NUM_DATA*TRAIN_RATIO)]
)

metadata=[]

# ============================================================
# Generate
# ============================================================

for i in range(NUM_DATA):

    try:

        # ----------------------------------------------------
        # Random scene
        # ----------------------------------------------------

        workers=randomize_scene()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        mujoco.mj_forward(
            model,
            data
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        renderer.update_scene(
            data,
            camera="dataset_camera"
        )

        img=renderer.render()

        img=cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        labels=[]

        # ----------------------------------------------------
        # Debug image
        # ----------------------------------------------------

        debug_img=img.copy() if DEBUG_POSE else None

        # ====================================================
        # Worker Pose
        # ====================================================

        for worker in workers:

            class_id=worker["class_id"]
            prefix=worker["geom_prefix"]

            # ------------------------------------------------
            # Get pose
            # ------------------------------------------------

            world_keypoints,image_keypoints=calculate_pose(
                prefix
            )

            # ------------------------------------------------
            # Visible keypoint count
            # ------------------------------------------------

            visible_count=sum(
                1
                for _,_,v in image_keypoints
                if v>0
            )

            if visible_count<MIN_VISIBLE_KEYPOINTS:
                continue

            # ------------------------------------------------
            # BBox
            # ------------------------------------------------

            bbox=bbox_from_keypoints(
                image_keypoints
            )

            if bbox is None:
                continue

            x1,y1,x2,y2=bbox

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------

            if DEBUG_POSE:

                draw_pose(
                    debug_img,
                    image_keypoints,
                    class_id
                )

                cv2.rectangle(
                    debug_img,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    debug_img,
                    CLASS_NAMES[class_id],
                    (x1,max(15,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0,255,0),
                    1,
                    cv2.LINE_AA
                )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            labels.append({
                "class_id":class_id,
                "bbox":bbox,
                "keypoints":image_keypoints
            })

        # ====================================================
        # Save path
        # ====================================================

        name=f"{i:06d}"

        if i in train_set:

            image_dir=TRAIN_IMAGE
            label_dir=TRAIN_LABEL

        else:

            image_dir=VAL_IMAGE
            label_dir=VAL_LABEL

        # ====================================================
        # Save image
        # ====================================================

        image_path=os.path.join(
            image_dir,
            f"{name}.jpg"
        )

        success=cv2.imwrite(
            image_path,
            img
        )

        if not success:

            raise RuntimeError(
                f"Failed to save image: {image_path}"
            )

        # ====================================================
        # Save debug
        # ====================================================

        if DEBUG_POSE:

            cv2.imwrite(
                os.path.join(
                    DEBUG_IMAGE_DIR,
                    f"{name}.jpg"
                ),
                debug_img
            )

        # ====================================================
        # Save YOLO Pose label
        # ====================================================

        label_path=os.path.join(
            label_dir,
            f"{name}.txt"
        )

        with open(
            label_path,
            "w"
        ) as f:

            for label in labels:

                cls=label["class_id"]

                x1,y1,x2,y2=label["bbox"]

                keypoints=label["keypoints"]

                # ------------------------------------------------
                # Normalize bbox
                # ------------------------------------------------

                cx=((x1+x2)/2)/WIDTH
                cy=((y1+y2)/2)/HEIGHT

                bw=(x2-x1)/WIDTH
                bh=(y2-y1)/HEIGHT

                values=[
                    str(cls),
                    f"{cx:.6f}",
                    f"{cy:.6f}",
                    f"{bw:.6f}",
                    f"{bh:.6f}"
                ]

                # ------------------------------------------------
                # Normalize keypoints
                # ------------------------------------------------

                for x,y,v in keypoints:

                    if v==0:

                        # invisible / unavailable
                        values.extend([
                            "0.000000",
                            "0.000000",
                            "0"
                        ])

                    else:

                        values.extend([
                            f"{x/WIDTH:.6f}",
                            f"{y/HEIGHT:.6f}",
                            str(v)
                        ])

                f.write(
                    " ".join(values)
                    +"\n"
                )

        # ====================================================
        # Metadata
        # ====================================================

        metadata.append({
            "image":name+".jpg",
            "workers":workers,
            "camera":model.cam_pos[cam_id].tolist(),
            "camera_quat":model.cam_quat[cam_id].tolist()
        })

        # ====================================================
        # Progress
        # ====================================================

        if i%100==0:

            print(
                f"{i}/{NUM_DATA}",
                flush=True
            )

    except Exception as e:

        print(
            f"\nERROR at sample {i}: {e}",
            flush=True
        )

        raise

# ============================================================
# Save metadata
# ============================================================

with open(
    META_PATH,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# Complete
# ============================================================

print()
print("YOLO Pose dataset generation complete!")
print(f"  Total : {NUM_DATA}")
print(f"  Train : {len(train_set)}")
print(f"  Val   : {NUM_DATA-len(train_set)}")
print(f"  Keypoints : {NUM_KEYPOINTS}")