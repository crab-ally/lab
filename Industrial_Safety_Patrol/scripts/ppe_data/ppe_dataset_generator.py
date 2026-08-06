#!/usr/bin/env python3

import mujoco
import numpy as np
import cv2
import os
import json
import random
import shutil

XML_PATH = "/workspace/worlds/ppe_dataset_world.xml"

# Dataset root
DATASET = "/workspace/datasets/ppe_dataset"

TRAIN_IMAGE = os.path.join(DATASET, "images/train")
VAL_IMAGE = os.path.join(DATASET, "images/val")

TRAIN_LABEL = os.path.join(DATASET, "labels/train")
VAL_LABEL = os.path.join(DATASET, "labels/val")

META_PATH = os.path.join(DATASET, "metadata.json")

for path in [
    TRAIN_IMAGE,
    VAL_IMAGE,
    TRAIN_LABEL,
    VAL_LABEL,
]:
    os.makedirs(path, exist_ok=True)

# Load MuJoCo
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Camera
cam_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "dataset_camera"
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
        value = c + np.random.uniform(-0.15,0.15)
        result.append(np.clip(value,0,1))

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

    u = WIDTH / 2 + f * x / depth
    v = HEIGHT / 2 - f * y / depth

    return (u, v)

################################################
# Collision Avoidance
################################################

def is_valid_position(x, y, radius=0.4):
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and (name.startswith("worker") or name == "floor"):
            continue
        
        pos = model.geom_pos[i]
        size = model.geom_size[i]
        gtype = model.geom_type[i]
        
        dx = abs(x - pos[0])
        dy = abs(y - pos[1])
        
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            if dx < size[0] + radius and dy < size[1] + radius:
                return False
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
            if np.hypot(dx, dy) < size[0] + radius:
                return False
    return True

################################################
# Occlusion check
################################################

def is_visible(point, worker_idx):
    
    pix = project(point)
    if pix is None:
        return False
    u, v = pix
    if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
        return False

    cam_pos = data.cam_xpos[cam_id]
    vec = point - cam_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return True
        
    vec_norm = vec / dist
    geomid = np.array([-1], dtype=np.int32)
    
    # Cast ray from camera to point
    hit_dist = mujoco.mj_ray(model, data, cam_pos, vec_norm, None, 1, -1, geomid)
    
    # If ray hits something before the point (with small margin)
    if hit_dist >= 0 and hit_dist < dist - 0.05:
        if geomid[0] != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])
            if name and name.startswith(f"worker_{worker_idx}_"):
                return True
        return False
    return True

################################################
# Sample geom surface
################################################

def sample_geom_points(gid, n_circle=24, n_height=8):

    center = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
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

                local = np.array([
                    r*c,
                    r*s,
                    z
                ])

                pts.append(center + R @ local)

    # Sphere
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:

        r = size[0]

        for phi in np.linspace(0, np.pi, 10):

            for theta in np.linspace(0, 2*np.pi, 20, endpoint=False):

                local = np.array([
                    r*np.sin(phi)*np.cos(theta),
                    r*np.sin(phi)*np.sin(theta),
                    r*np.cos(phi)
                ])

                pts.append(center + R @ local)

    # Box
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:

        sx, sy, sz = size

        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):

                    local = np.array([
                        dx,
                        dy,
                        dz
                    ])

                    pts.append(center + R @ local)

    # 기타 geom은 중심점만
    else:

        pts.append(center)

    return pts

################################################
# Visible ratio
################################################

def visible_ratio(names, worker_idx):

    visible = 0
    total = 0

    for name in names:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name
        )

        if gid < 0:
            continue

        for p in sample_geom_points(gid):

            total += 1

            if is_visible(p, worker_idx):
                visible += 1

    if total == 0:
        return 0.0

    return visible / total

################################################
# Generic BBox
################################################

def bbox_from_geoms(names, worker_idx):

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

            if not is_visible(p, worker_idx):
                continue

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

        for _ in range(100):
            x=np.random.uniform(-4.5,4.5)
            y=np.random.uniform(-4.5,4.5)
            if is_valid_position(x, y, 0.4):
                break
        model.body_pos[bid]=[x,y,0]

        ################################################
        # rotation
        ################################################

        yaw=np.random.uniform(0,2*np.pi)
        model.body_quat[bid]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]

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

    # 카메라 높이
    CAM_Z = 0.383

    # 먼저 바라볼 작업자 선택
    target = random.choice(workers)

    # 최대 100번 시도
    for _ in range(100):

        cam_x = np.random.uniform(-4.5, 4.5)
        cam_y = np.random.uniform(-4.5, 4.5)

        if not is_valid_position(cam_x, cam_y, 0.3):
            continue

        dist = np.hypot(
            target["x"] - cam_x,
            target["y"] - cam_y
        )

        if 2.0 <= dist <= 6.0:
            break

    # 100번 모두 실패하면 마지막 위치 사용
    model.cam_pos[cam_id] = [cam_x, cam_y, CAM_Z]

    # 작업자를 바라보는 방향 계산
    dx = target["x"] - cam_x
    dy = target["y"] - cam_y

    yaw = np.arctan2(dy, dx)

    # ±25도 랜덤
    yaw += np.random.uniform(
        np.deg2rad(-25),
        np.deg2rad(25)
    )

    base_quat = np.array([0.5, 0.5, -0.5, -0.5])

    yaw_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(
        yaw_quat,
        np.array([0, 0, 1]),
        yaw
    )

    result = np.zeros(4)
    mujoco.mju_mulQuat(
        result,
        yaw_quat,
        base_quat
    )

    model.cam_quat[cam_id] = result



    return workers

################################################
# Generate Dataset
################################################

NUM_DATA=100

TRAIN_RATIO = 0.8

indices = list(range(NUM_DATA))
random.seed(42)
random.shuffle(indices)

train_set = set(indices[:int(NUM_DATA * TRAIN_RATIO)])

metadata=[]

for i in range(NUM_DATA):

    workers=randomize_scene()

    mujoco.mj_forward(model,data)

    renderer.update_scene(
        data,
        camera="dataset_camera"
    )

    img=renderer.render()
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ################################################
    # Debug projection (All workers)
    ################################################

    for worker in workers:

        idx = worker["id"]

        parts = [
            f"worker_{idx}_body",
            f"worker_{idx}_head",
        ]

        if worker["helmet"]:
            parts.append(f"worker_{idx}_helmet")

        if worker["vest"]:
            parts.append(f"worker_{idx}_vest")
        
        ################################################
        # Label과 동일한 visibility 조건
        ################################################

        head_ratio = visible_ratio(
            [f"worker_{idx}_head"],
            idx
        )

        if head_ratio < 0.5:
            continue

        body_ratio = visible_ratio(
            [f"worker_{idx}_body"],
            idx
        )

        if body_ratio < 0.5:
            continue

        ################################################
        # 1. Draw geom centers (Red)
        ################################################

        for geom_name in parts:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name
            )

            if gid < 0:
                continue

            world = data.geom_xpos[gid]

            # 가려져 있으면 표시하지 않음
            if not is_visible(world, idx):
                continue

            pix = project(world)

            if pix is None:
                continue

            u, v = map(int, pix)

            if 0 <= u < WIDTH and 0 <= v < HEIGHT:

                cv2.circle(
                    img,
                    (u, v),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    img,
                    geom_name,
                    (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

        ################################################
        # 2. Draw sampled surface points (Yellow)
        ################################################

        for geom_name in parts:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name
            )

            if gid < 0:
                continue

            for world in sample_geom_points(gid):

                if not is_visible(world, idx):
                    continue

                pix = project(world)

                if pix is None:
                    continue

                u, v = map(int, pix)

                if 0 <= u < WIDTH and 0 <= v < HEIGHT:

                    cv2.circle(
                        img,
                        (u, v),
                        2,
                        (0, 255, 255),
                        -1
                    )
        ################################################
        # 3. Draw bounding box (Blue)
        ################################################

        bbox = bbox_from_geoms(parts, idx)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        ################################################
        # 4. Draw bbox corners (Magenta)
        ################################################

        cv2.circle(img, (x1, y1), 5, (255, 0, 255), -1)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), -1)

        cv2.putText(
            img,
            f"W{idx}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )

    labels=[]

    ############################################
    # labels
    ############################################

    for worker in workers:

        idx=worker["id"]

        ################################################
        # Skip if head or body is not visible
        ################################################

        head_ratio = visible_ratio(
            [f"worker_{idx}_head"],
            idx
        )

        if head_ratio < 0.5:
            continue

        body_ratio = visible_ratio(
            [f"worker_{idx}_body"],
            idx
        )

        if body_ratio < 0.5:
            continue

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
        bbox=bbox_from_geoms(parts, idx)

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
                ], idx
            )
            if helmet:
                labels.append(
                    (1, helmet)
                )

        if worker["vest"]:
            vest=bbox_from_geoms(
                [
                    f"worker_{idx}_vest"
                ], idx
            )
            if vest:
                labels.append(
                    (2, vest)
                )

    ############################################
    # Save image
    ############################################

    name = f"{i:06d}"

    if i in train_set:
        image_dir = TRAIN_IMAGE
        label_dir = TRAIN_LABEL
    else:
        image_dir = VAL_IMAGE
        label_dir = VAL_LABEL

    cv2.imwrite(
        os.path.join(image_dir, f"{name}.jpg"),
        img
    )
    ############################################
    # Save YOLO
    ############################################

    with open(
        os.path.join(label_dir, f"{name}.txt"),
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
            "camera_quat":model.cam_quat[cam_id].tolist()
        }
    )

    if i%100==0:
        print(f"{i}/{NUM_DATA}")

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

print("Dataset generation complete")