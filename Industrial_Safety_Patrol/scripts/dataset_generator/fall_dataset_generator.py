#!/usr/bin/env python3
"""
Fall Detection 데이터셋 생성기

클래스:
0: fallen
1: standing

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

XML_PATH = "/workspace/worlds/dataset/fall_dataset_world.xml"

DATASET = "/workspace/datasets/fall_dataset"

TRAIN_IMAGE = os.path.join(DATASET, "images/train")
VAL_IMAGE   = os.path.join(DATASET, "images/val")

TRAIN_LABEL = os.path.join(DATASET, "labels/train")
VAL_LABEL   = os.path.join(DATASET, "labels/val")

META_PATH = os.path.join(DATASET, "metadata.json")


# ============================================================
# Debug
# ============================================================

DEBUG_BBOX = False
DEBUG_IMAGE_DIR = os.path.join(DATASET, "debug")

DEBUG_COLORS = {
    0: (0, 0, 255),       # fallen = red
    1: (0, 220, 50),      # standing = green
}

DEBUG_CLASS_NAMES = {
    0: "fallen",
    1: "standing",
}

# ============================================================
# Dataset directories
# ============================================================

for path in [
    TRAIN_IMAGE,
    VAL_IMAGE,
    TRAIN_LABEL,
    VAL_LABEL,
]:
    os.makedirs(path, exist_ok=True)

if DEBUG_BBOX:
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


# ============================================================
# Load MuJoCo
# ============================================================

print("Loading MuJoCo model...")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

cam_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "dataset_camera"
)

if cam_id < 0:
    raise RuntimeError("dataset_camera not found")

print("MuJoCo model loaded")

# ============================================================
# Workers
# ============================================================

NUM_FALLEN = 5
NUM_STANDING = 3

fallen_ids = []
standing_ids = []


for i in range(NUM_FALLEN):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"fallen_worker_{i}"
    )

    if bid < 0:
        raise RuntimeError(
            f"fallen_worker_{i} not found"
        )

    fallen_ids.append(bid)


for i in range(NUM_STANDING):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"standing_worker_{i}"
    )

    if bid < 0:
        raise RuntimeError(
            f"standing_worker_{i} not found"
        )

    standing_ids.append(bid)

# ============================================================
# Renderer
# ============================================================

WIDTH = 640
HEIGHT = 480

renderer = mujoco.Renderer(
    model,
    HEIGHT,
    WIDTH
)

# ============================================================
# Random color
# ============================================================

def random_color(base):

    result = []

    for c in base:
        value = c + np.random.uniform(-0.15, 0.15)
        result.append(np.clip(value, 0, 1))

    return result

# ============================================================
# World -> Pixel
# ============================================================

def project(point):

    cam_pos = data.cam_xpos[cam_id]

    R = data.cam_xmat[cam_id].reshape(3, 3)

    # world -> camera
    pc = R.T @ (
        np.asarray(point) - cam_pos
    )

    x, y, z = pc

    # MuJoCo camera looks along -Z
    depth = -z

    if depth <= 1e-6:
        return None

    fovy = model.cam_fovy[cam_id]

    f = HEIGHT / (
        2 *
        np.tan(
            np.deg2rad(fovy) / 2
        )
    )

    u = WIDTH / 2 + f * x / depth
    v = HEIGHT / 2 - f * y / depth

    return (u, v)

# ============================================================
# Collision avoidance
# ============================================================

def is_valid_position(
    x,
    y,
    radius=0.5
):

    for i in range(model.ngeom):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            i
        )

        if name and (
            name.startswith("fallen_worker")
            or name.startswith("standing_worker")
            or name == "floor"
        ):
            continue

        pos = model.geom_pos[i]
        size = model.geom_size[i]
        gtype = model.geom_type[i]

        dx = abs(x - pos[0])
        dy = abs(y - pos[1])

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:

            if (
                dx < size[0] + radius
                and
                dy < size[1] + radius
            ):
                return False

        elif gtype in (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_SPHERE
        ):

            if np.hypot(dx, dy) < size[0] + radius:
                return False

    return True

# ============================================================
# Visibility
# ============================================================

def is_visible(point, owner_prefix):

    point = np.asarray(point)

    pix = project(point)

    if pix is None:
        return False

    u, v = pix

    if not (
        0 <= u < WIDTH
        and
        0 <= v < HEIGHT
    ):
        return False

    cam_pos = data.cam_xpos[cam_id]

    vec = point - cam_pos

    dist = np.linalg.norm(vec)

    if dist < 1e-6:
        return True

    vec_norm = vec / dist

    geomid = np.array(
        [-1],
        dtype=np.int32
    )

    hit_dist = mujoco.mj_ray(
        model,
        data,
        cam_pos,
        vec_norm,
        None,
        1,
        -1,
        geomid
    )

    if hit_dist >= 0 and hit_dist < dist - 0.05:

        if geomid[0] != -1:

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geomid[0]
            )

            if (
                name
                and
                name.startswith(owner_prefix)
            ):
                return True

        return False

    return True

# ============================================================
# Sample geom surface
# ============================================================

def sample_geom_points(
    gid,
    n_circle=32,
    n_height=16,
    n_phi=16,
    n_theta=32
):

    center = data.geom_xpos[gid]

    R = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]

    gtype = model.geom_type[gid]

    pts = []

    # Sphere
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:

        r = size[0]

        for phi in np.linspace(
            0,
            np.pi,
            n_phi
        ):

            for theta in np.linspace(
                0,
                2 * np.pi,
                n_theta,
                endpoint=False
            ):

                local = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])

                pts.append(
                    center + R @ local
                )

    # Capsule
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:

        r = size[0]
        half_length = size[1]

        # Cylinder surface

        for z in np.linspace(
            -half_length,
            half_length,
            n_height
        ):

            for theta in np.linspace(
                0,
                2 * np.pi,
                n_circle,
                endpoint=False
            ):

                local = np.array([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    z
                ])

                pts.append(
                    center + R @ local
                )

        # Top hemisphere
        for phi in np.linspace(
            0,
            np.pi / 2,
            n_phi // 2
        ):

            z = (
                half_length
                +
                r * np.cos(phi)
            )

            rr = r * np.sin(phi)

            for theta in np.linspace(
                0,
                2 * np.pi,
                n_circle,
                endpoint=False
            ):

                local = np.array([
                    rr * np.cos(theta),
                    rr * np.sin(theta),
                    z
                ])

                pts.append(
                    center + R @ local
                )

        # Bottom hemisphere
        for phi in np.linspace(
            0,
            np.pi / 2,
            n_phi // 2
        ):

            z = (
                -half_length
                -
                r * np.cos(phi)
            )

            rr = r * np.sin(phi)

            for theta in np.linspace(
                0,
                2 * np.pi,
                n_circle,
                endpoint=False
            ):

                local = np.array([
                    rr * np.cos(theta),
                    rr * np.sin(theta),
                    z
                ])

                pts.append(
                    center + R @ local
                )

    # Box
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:

        sx, sy, sz = size

        values_x = np.linspace(-sx, sx, 8)
        values_y = np.linspace(-sy, sy, 8)
        values_z = np.linspace(-sz, sz, 8)

        # X faces
        for x in [-sx, sx]:

            for y in values_y:

                for z in values_z:

                    local = np.array([
                        x,
                        y,
                        z
                    ])

                    pts.append(center + R @ local)

        # Y faces
        for y in [-sy, sy]:

            for x in values_x:

                for z in values_z:

                    local = np.array([
                        x,
                        y,
                        z
                    ])

                    pts.append(center + R @ local)

        # Z faces
        for z in [-sz, sz]:

            for x in values_x:

                for y in values_y:

                    local = np.array([
                        x,
                        y,
                        z
                    ])

                    pts.append(center + R @ local)

    # Cylinder
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:

        r = size[0]
        half_length = size[1]

        for z in np.linspace(
            -half_length,
            half_length,
            n_height
        ):

            for theta in np.linspace(
                0,
                2 * np.pi,
                n_circle,
                endpoint=False
            ):

                local = np.array([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    z
                ])

                pts.append(center + R @ local)

        # top / bottom center
        pts.append(center + R[:, 2] * half_length)
        pts.append(center - R[:, 2] * half_length)

    else:
        pts.append(center)

    return pts

# ============================================================
# Visible ratio
# ============================================================

def visible_ratio(names, owner_prefix):

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

        points = sample_geom_points(gid)

        for p in points:

            total += 1

            if is_visible(
                p,
                owner_prefix
            ):
                visible += 1

    if total == 0:
        return 0.0

    return visible / total

# ============================================================
# BBox from visible geometry
# ============================================================

def bbox_from_geoms(names, owner_prefix):

    pixels = []

    for name in names:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name
        )

        if gid < 0:
            continue

        points = sample_geom_points(gid)

        for p in points:

            # Only visible surface points
            if not is_visible(
                p,
                owner_prefix
            ):
                continue

            pix = project(p)

            if pix is not None:
                pixels.append(pix)


    if len(pixels) < 20:
        return None


    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]

    x1 = np.clip(min(xs), 0, WIDTH - 1)
    y1 = np.clip(min(ys), 0, HEIGHT - 1)

    x2 = np.clip(max(xs), 0, WIDTH - 1)
    y2 = np.clip(max(ys), 0, HEIGHT - 1)


    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < 6:
        return None

    if (y2 - y1) < 6:
        return None


    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )

# Worker body parts
BODY_PARTS = [
    "torso",
    "pelvis",
    "luleg",
    "llleg",
    "ruleg",
    "rlleg",
    "larm",
    "rarm",
    "head",
]

# ============================================================
# Fallen quaternion
# ============================================================

def make_fallen_quat(yaw):

    tilt_quat = np.zeros(4)

    mujoco.mju_axisAngle2Quat(
        tilt_quat,
        np.array([
            1.0,
            0.0,
            0.0
        ]),
        np.pi / 2
    )

    yaw_quat = np.zeros(4)

    mujoco.mju_axisAngle2Quat(
        yaw_quat,
        np.array([
            0.0,
            0.0,
            1.0
        ]),
        yaw
    )

    result = np.zeros(4)

    mujoco.mju_mulQuat(
        result,
        yaw_quat,
        tilt_quat
    )

    return result

# ============================================================
# Random Scene
# ============================================================

def randomize_scene():

    workers = []

    # ========================================================
    # Fallen workers
    # ========================================================

    n_fallen = random.randint(
        1,
        NUM_FALLEN
    )


    for i, bid in enumerate(
        fallen_ids
    ):

        if i >= n_fallen:

            # Hide
            model.body_pos[bid] = [
                100.0,
                100.0,
                -10.0
            ]

            continue


        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        x = 0.0
        y = 0.0

        for _ in range(100):

            x = np.random.uniform(
                -4.0,
                4.0
            )

            y = np.random.uniform(
                -4.0,
                4.0
            )

            if is_valid_position(
                x,
                y,
                0.7
            ):
                break


        # ----------------------------------------------------
        # Fallen rotation
        # ----------------------------------------------------

        yaw = np.random.uniform(0, 2 * np.pi)
        fallen_quat = make_fallen_quat(yaw)

        model.body_pos[bid] = [x, y, 0.06]
        model.body_quat[bid] = fallen_quat

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = random_color(
            np.array([
                0.2,
                0.24,
                0.28
            ])
        )


        for part in BODY_PARTS:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"fw{i}_{part}"
            )

            if gid >= 0:

                model.geom_rgba[
                    gid
                ][:3] = color


        workers.append({
            "id": i,
            "type": "fallen",
            "x": float(x),
            "y": float(y),
        })


    # ========================================================
    # Standing workers
    # ========================================================

    n_standing = random.randint(
        0,
        NUM_STANDING
    )


    for i, bid in enumerate(
        standing_ids
    ):

        if i >= n_standing:

            # Hide
            model.body_pos[bid] = [
                100.0,
                100.0,
                -10.0
            ]

            continue

        x = 0.0
        y = 0.0


        for _ in range(100):

            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)

            if is_valid_position(
                x,
                y,
                0.5
            ):
                break

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        yaw = np.random.uniform(0, 2 * np.pi)
        model.body_pos[bid] = [x, y, 0.0]

        model.body_quat[bid] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2)
        ]

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = random_color(
            np.array([
                0.2,
                0.24,
                0.28
            ])
        )

        for part in BODY_PARTS:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"sw{i}_{part}"
            )

            if gid >= 0:

                model.geom_rgba[
                    gid
                ][:3] = color


        workers.append({
            "id": i,
            "type": "standing",
            "x": float(x),
            "y": float(y),
        })

    # ========================================================
    # Camera
    # ========================================================

    CAM_Z = 0.383

    fallen_workers = [
        w
        for w in workers
        if w["type"] == "fallen"
    ]

    if not fallen_workers:
        raise RuntimeError(
            "No fallen worker exists"
        )

    target = random.choice(
        fallen_workers
    )

    cam_x = target["x"]
    cam_y = target["y"]

    for _ in range(100):

        candidate_x = np.random.uniform(-4.0, 4.0)
        candidate_y = np.random.uniform(-4.0, 4.0)

        if not is_valid_position(
            candidate_x,
            candidate_y,
            0.3
        ):
            continue

        dist = np.hypot(
            target["x"] - candidate_x,
            target["y"] - candidate_y
        )

        if 2.0 <= dist <= 7.0:

            cam_x = candidate_x
            cam_y = candidate_y

            break

    model.cam_pos[cam_id] = [
        cam_x,
        cam_y,
        CAM_Z
    ]

    # --------------------------------------------------------
    # Camera direction
    # --------------------------------------------------------

    dx = target["x"] - cam_x
    dy = target["y"] - cam_y

    yaw = np.arctan2(dy, dx)
    yaw += np.random.uniform(np.deg2rad(-20), np.deg2rad(20))

    base_quat = np.array([0.5, 0.5, -0.5, -0.5])
    yaw_quat = np.zeros(4)

    mujoco.mju_axisAngle2Quat(
        yaw_quat,
        np.array([
            0,
            0,
            1
        ]),
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

# ============================================================
# Dataset generation
# ============================================================

NUM_DATA = 8000
TRAIN_RATIO = 0.8

CLASS_FALLEN = 0
CLASS_STANDING = 1


indices = list(range(NUM_DATA))

random.seed(42)
np.random.seed(42)

random.shuffle(indices)

train_set = set(
    indices[
        :int(NUM_DATA * TRAIN_RATIO)
    ]
)

metadata = []

# ============================================================
# Generate
# ============================================================

for i in range(NUM_DATA):

    print(
        f"[{i + 1}/{NUM_DATA}] generating...",
        flush=True
    )

    try:

        # Random scene
        workers = randomize_scene()

        # Forward
        mujoco.mj_forward(model, data)

        # Render
        renderer.update_scene(data, camera="dataset_camera")
        img = renderer.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        labels = []

        # ====================================================
        # Worker labels
        # ====================================================

        for worker in workers:

            idx = worker["id"]

            worker_type = worker["type"]

            # Prefix
            if worker_type == "fallen":

                prefix = f"fw{idx}_"
                class_id = CLASS_FALLEN

            else:

                prefix = f"sw{idx}_"
                class_id = CLASS_STANDING

            # Body parts
            parts = [
                f"{prefix}torso",
                f"{prefix}pelvis",
                f"{prefix}luleg",
                f"{prefix}llleg",
                f"{prefix}ruleg",
                f"{prefix}rlleg",
                f"{prefix}larm",
                f"{prefix}rarm",
                f"{prefix}head",
            ]


            # Visibility
            head_ratio = visible_ratio(
                [f"{prefix}head"],
                prefix
            )

            if head_ratio < 0.5:
                continue

            body_parts = [
                f"{prefix}torso",
                f"{prefix}pelvis",
                f"{prefix}luleg",
                f"{prefix}llleg",
                f"{prefix}ruleg",
                f"{prefix}rlleg",
                f"{prefix}larm",
                f"{prefix}rarm",
            ]

            body_ratio = visible_ratio(
                body_parts,
                prefix
            )

            if body_ratio < 0.5:
                continue

            # Worker BBox
            bbox = bbox_from_geoms(
                parts,
                prefix
            )

            if bbox is not None:

                labels.append((class_id, bbox))

        # Save path
        name = f"{i:06d}"

        if i in train_set:

            image_dir = TRAIN_IMAGE
            label_dir = TRAIN_LABEL

        else:

            image_dir = VAL_IMAGE
            label_dir = VAL_LABEL

        # Save image
        image_path = os.path.join(image_dir, f"{name}.jpg")
        success = cv2.imwrite(image_path, img)

        if not success:

            raise RuntimeError(
                f"Failed to save image: {image_path}"
            )

        # Debug bbox
        if DEBUG_BBOX:

            debug_img = img.copy()

            for cls, bbox in labels:

                x1, y1, x2, y2 = bbox

                color = DEBUG_COLORS.get(
                    cls,
                    (255, 255, 0)
                )

                cv2.rectangle(
                    debug_img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label_text = DEBUG_CLASS_NAMES.get(
                    cls,
                    str(cls)
                )

                (
                    tw,
                    th
                ), _ = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    1
                )

                cv2.rectangle(
                    debug_img,
                    (
                        x1,
                        max(
                            y1 - th - 6,
                            0
                        )
                    ),
                    (
                        x1 + tw + 6,
                        max(
                            y1,
                            th + 6
                        )
                    ),
                    color,
                    -1
                )

                cv2.putText(
                    debug_img,
                    label_text,
                    (
                        x1 + 3,
                        max(
                            y1 - 4,
                            th + 2
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            cv2.imwrite(
                os.path.join(
                    DEBUG_IMAGE_DIR,
                    f"{name}.jpg"
                ),
                debug_img
            )

        # Save YOLO label
        label_path = os.path.join(label_dir, f"{name}.txt")

        with open(label_path, "w") as f:

            for cls, bbox in labels:

                x1, y1, x2, y2 = bbox

                x1 = np.clip(x1, 0, WIDTH - 1)
                y1 = np.clip(y1, 0, HEIGHT - 1)
                x2 = np.clip(x2, 0, WIDTH - 1)
                y2 = np.clip(y2, 0, HEIGHT - 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                if (x2 - x1) < 6:
                    continue

                if (y2 - y1) < 6:
                    continue

                cx = ((x1 + x2) / 2) / WIDTH
                cy = ((y1 + y2) / 2) / HEIGHT

                bw = ((x2 - x1) / WIDTH)
                bh = ((y2 - y1) / HEIGHT)

                if bw < 0.01 or bh < 0.01:
                    continue

                f.write(
                    f"{cls} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}\n"
                )

        # Metadata
        metadata.append({
            "image": name + ".jpg",
            "workers": workers,
            "camera": model.cam_pos[
                cam_id
            ].tolist(),
            "camera_quat": model.cam_quat[
                cam_id
            ].tolist(),
        })

        # Progress
        if i%100==0:
            print(f"{i}/{NUM_DATA}")


    except Exception as e:

        print(
            f"\nERROR at sample {i}: {e}",
            flush=True
        )

        raise


# Save metadata
with open(META_PATH, "w") as f:
    json.dump(metadata,f,indent=4)


# Complete
print()
print("Fall dataset generation complete!")
print(f"  Total : {NUM_DATA}")
print(f"  Train : {len(train_set)}")
print(f"  Val   : {NUM_DATA - len(train_set)}")