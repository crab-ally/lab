#!/usr/bin/env python3

import mujoco
import numpy as np
import cv2
import os
import json
import random

# Path
XML_PATH = "/workspace/worlds/ppe_dataset_world.xml"
IMAGE_DIR = "/workspace/datasets/ppe_dataset/images"
LABEL_DIR = "/workspace/datasets/ppe_dataset/labels"
META_PATH = "/workspace/datasets/ppe_dataset/metadata.json"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Load MuJoCo
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Camera
cam_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "dataset_camera"
)

# Light
light_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_LIGHT,
    "dataset_light"
)

################################################
# Workers
################################################

NUM_WORKERS = 5

worker_ids=[]

for i in range(NUM_WORKERS):

    bid=mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"worker_{i}"
    )

    worker_ids.append(bid)

################################################
# Renderer
################################################

WIDTH=640
HEIGHT=480

renderer=mujoco.Renderer(
    model,
    HEIGHT,
    WIDTH
)

################################################
# Color random
################################################

def random_color(base):

    result=[]

    for c in base:

        value=c+np.random.uniform(
            -0.15,
            0.15
        )

        result.append(
            np.clip(value,0,1)
        )

    return result

################################################
# World -> Pixel
################################################

def project(point):

    cam_pos = data.cam_xpos[cam_id]

    # camera rotation
    R = data.cam_xmat[cam_id].reshape(3, 3)

    # world -> camera
    pc = R.T @ (np.asarray(point) - cam_pos)

    x, y, z = pc

    depth = -z  # 카메라는 -Z 방향을 바라봄
    if depth <= 1e-6:
        return None

    fovy = model.cam_fovy[cam_id]
    f = HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))

    u = f * x / z + WIDTH / 2

    # OpenCV y축
    v = HEIGHT / 2 - f * y / z

    return (u, v)

################################################
# Sample geom surface
################################################

def sample_geom_points(gid, n_circle=24, n_height=8):

    center = data.geom_xpos[gid]
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]

    pts = []

    # Cylinder
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:

        r = size[0]
        h = size[1]

        for theta in np.linspace(0, 2*np.pi, n_circle, endpoint=False):

            c = np.cos(theta)
            s = np.sin(theta)

            for z in np.linspace(-h, h, n_height):

                pts.append(center + np.array([
                    r*c,
                    r*s,
                    z
                ]))

    # Sphere
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:

        r = size[0]

        for phi in np.linspace(0, np.pi, 10):

            for theta in np.linspace(0, 2*np.pi, 20, endpoint=False):

                pts.append(center + np.array([
                    r*np.sin(phi)*np.cos(theta),
                    r*np.sin(phi)*np.sin(theta),
                    r*np.cos(phi)
                ]))

    # Box
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:

        sx, sy, sz = size

        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):

                    pts.append(center + np.array([
                        dx,
                        dy,
                        dz
                    ]))

    # 기타 geom은 중심점만
    else:

        pts.append(center)

    return pts


################################################
# Generic BBox
################################################

def bbox_from_geoms(names):

    pixels=[]


    for name in names:


        gid=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name
        )


        if gid < 0:
            continue



        for p in sample_geom_points(gid):

            pix=project(p)

            if pix is not None:
                pixels.append(pix)



    if len(pixels)<20:
        return None



    xs=[p[0] for p in pixels]
    ys=[p[1] for p in pixels]



    x1=np.clip(min(xs),0,WIDTH-1)
    y1=np.clip(min(ys),0,HEIGHT-1)

    x2=np.clip(max(xs),0,WIDTH-1)
    y2=np.clip(max(ys),0,HEIGHT-1)



    if x2 <= x1 or y2 <= y1:
        return None



    if (x2-x1)<6 or (y2-y1)<6:
        return None



    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )

################################################
# Random Scene
################################################

def randomize_scene():


    workers=[]



    ################################################
    # number of people
    ################################################

    count=random.randint(
        1,
        NUM_WORKERS
    )



    for i,bid in enumerate(worker_ids):


        if i >= count:


            # hide

            model.body_pos[bid]=[
                100,
                100,
                -10
            ]


            continue




        ################################################
        # position
        ################################################


        x=np.random.uniform(
            -5,
            5
        )

        y=np.random.uniform(
            -5,
            5
        )


        model.body_pos[bid]=[
            x,
            y,
            0
        ]




        ################################################
        # rotation
        ################################################

        yaw=np.random.uniform(
            0,
            2*np.pi
        )


        model.body_quat[bid]=[

            np.cos(yaw/2),

            0,

            0,

            np.sin(yaw/2)

        ]




        ################################################
        # PPE
        ################################################

        has_helmet=random.choice(
            [
                True,
                False
            ]
        )

        has_vest=random.choice(
            [
                True,
                False
            ]
        )



        workers.append(
            {
                "id":i,
                "x":x,
                "y":y,
                "helmet":has_helmet,
                "vest":has_vest
            }
        )

        ################################################
        # Color variation
        ################################################

        body_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"worker_{i}_body"
        )


        model.geom_rgba[body_id][:3]=random_color(
            np.array(
                [
                    0.2,
                    0.25,
                    0.3
                ]
            )
        )



        ################################################
        # helmet / vest alpha
        ################################################

        for part in ["helmet","vest"]:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"worker_{i}_{part}"
            )

            if gid < 0:
                continue

            if part=="helmet":
                visible=has_helmet
            elif part=="vest":
                visible=has_vest

            model.geom_rgba[gid][3] = 1 if visible else 0

    ################################################
    # camera pose (robot mounted camera simulation)
    ################################################


    # 카메라 높이 고정
    CAM_Z = 0.383

    # 위치만 랜덤
    cam_x = np.random.uniform(-4,4)
    cam_y = np.random.uniform(-4,4)
    model.cam_pos[cam_id]=[cam_x, cam_y, CAM_Z]

    # 기본 카메라 설치 방향
    base_quat = np.array([0.5,0.5,-0.5,-0.5])

    # 로봇 yaw 방향 변화
    yaw = np.random.uniform(-np.pi,np.pi)

    yaw_quat = np.zeros(4)

    mujoco.mju_axisAngle2Quat(yaw_quat,np.array([0,0,1]),yaw)

    # yaw rotation * camera base rotation
    result = np.zeros(4)

    mujoco.mju_mulQuat(
        result,
        yaw_quat,
        base_quat
    )

    model.cam_quat[cam_id]=result

    ################################################
    # light
    ################################################


    model.light_pos[light_id]=[

        np.random.uniform(-5,5),

        np.random.uniform(-5,5),

        np.random.uniform(3,7)

    ]



    return workers





################################################
# Generate Dataset
################################################

NUM_DATA=3000


metadata=[]



for i in range(NUM_DATA):


    workers=randomize_scene()



    mujoco.mj_forward(
        model,
        data
    )



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



    ############################################
    # labels
    ############################################


    for worker in workers:

        idx=worker["id"]

        # person
        parts=[
            f"worker_{idx}_body",
            f"worker_{idx}_head"
        ]
        if worker["helmet"]:
            parts.append(
                f"worker_{idx}_helmet"
            )
        if worker["vest"]:
            parts.append(
                f"worker_{idx}_vest"
            )
        bbox=bbox_from_geoms(parts)

        if bbox:
            labels.append(
                (
                    0,
                    bbox
                )
            )

        if worker["helmet"]:
            helmet=bbox_from_geoms(
                [
                    f"worker_{idx}_helmet"
                ]
            )
            if helmet:
                labels.append(
                    (1, helmet)
                )

        if worker["vest"]:
            vest=bbox_from_geoms(
                [
                    f"worker_{idx}_vest"
                ]
            )
            if vest:
                labels.append(
                    (2, vest)
                )

    ############################################
    # Save image
    ############################################


    name=f"{i:06d}"



    cv2.imwrite(

        f"{IMAGE_DIR}/{name}.jpg",

        img

    )



    ############################################
    # Save YOLO
    ############################################


    with open(

        f"{LABEL_DIR}/{name}.txt",

        "w"

    ) as f:


        for cls,bbox in labels:


            x1, y1, x2, y2 = bbox

            x1 = np.clip(x1, 0, WIDTH - 1)
            y1 = np.clip(y1, 0, HEIGHT - 1)
            x2 = np.clip(x2, 0, WIDTH - 1)
            y2 = np.clip(y2, 0, HEIGHT - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            if (x2-x1)<6 or (y2-y1)<6:
                continue

            cx = ((x1 + x2) / 2) / WIDTH
            cy = ((y1 + y2) / 2) / HEIGHT
            bw = (x2 - x1) / WIDTH
            bh = (y2 - y1) / HEIGHT

            if bw < 0.01 or bh < 0.01:
                continue

            f.write(

                f"{cls} "

                f"{cx:.6f} "

                f"{cy:.6f} "

                f"{bw:.6f} "

                f"{bh:.6f}\n"

            )



    ############################################
    # metadata
    ############################################


    metadata.append(
        {
            "image":name+".jpg",
            "workers":workers,
            "camera":model.cam_pos[cam_id].tolist(),
            "camera_quat":model.cam_quat[cam_id].tolist(),
            "light":model.light_pos[light_id].tolist()
        }
    )



    if i%100==0:

        print(
            f"{i}/{NUM_DATA}"
        )





################################################
# save metadata
################################################


with open(
    META_PATH,
    "w"
) as f:


    json.dump(
        metadata,
        f,
        indent=4
    )



print(
    "Dataset generation complete"
)