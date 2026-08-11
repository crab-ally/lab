#!/usr/bin/env python3
"""
NUM_DATA
"""
import mujoco
import numpy as np
import cv2
import os
import json
import random
import shutil

XML_PATH = "/workspace/worlds/dataset/ppe_forklift_dataset_world.xml"
DATASET = "/workspace/datasets/ppe_forklift_dataset"

TRAIN_IMAGE = os.path.join(DATASET, "images/train")
VAL_IMAGE = os.path.join(DATASET, "images/val")
TRAIN_LABEL = os.path.join(DATASET, "labels/train")
VAL_LABEL = os.path.join(DATASET, "labels/val")
META_PATH = os.path.join(DATASET, "metadata.json")

DEBUG_BBOX = False
DEBUG_IMAGE_DIR = os.path.join(DATASET, "debug")
DEBUG_COLORS = {
    0: (220, 80, 0),
    1: (0, 200, 255),
    2: (0, 220, 50),
    3: (0, 128, 255),
}
DEBUG_CLASS_NAMES = {0: "person", 1: "helmet", 2: "vest", 3: "forklift"}

for path in [TRAIN_IMAGE, VAL_IMAGE, TRAIN_LABEL, VAL_LABEL]:
    os.makedirs(path, exist_ok=True)
if DEBUG_BBOX:
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "dataset_camera")

################################################
# Workers
################################################

WORKER_SPECS = [
    ("fallen_faceup_0", "ffu0", False),
    ("fallen_facedown_0", "ffd0", False),
    ("standing_worker_0", "sw0", False),
    ("standing_worker_1", "sw1", False),
    ("bending_worker_0", "bw0", False),
    ("bending_worker_1", "bw1", False),
    ("sitting_worker_0", "stw0", True),
    ("sitting_worker_1", "stw1", True),
]

workers_info = []

for body_name, geom_prefix, is_sitting in WORKER_SPECS:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Worker body not found: {body_name}")

    body_parts = ["torso", "pelvis", "luleg", "llleg", "ruleg", "rlleg", "larm", "rarm"]
    if is_sitting:
        body_parts = ["torso", "pelvis", "lthigh", "llleg", "rthigh", "rlleg", "larm", "rarm"]

    workers_info.append({
        "name": body_name,
        "id": bid,
        "geom_prefix": geom_prefix,
        "is_sitting": is_sitting,
        "body_parts": body_parts,
        "initial_pos": np.array(model.body_pos[bid], dtype=np.float64),
        "initial_quat": np.array(model.body_quat[bid], dtype=np.float64),
    })

################################################
# Renderer
################################################

WIDTH = 640
HEIGHT = 480
renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

################################################
# Color random
################################################

def random_color(base):
    result = []
    for c in base:
        value = c + np.random.uniform(-0.15, 0.15)
        result.append(np.clip(value, 0, 1))
    return result

################################################
# World -> Pixel
################################################

def project(point):
    cam_pos = data.cam_xpos[cam_id]
    R = data.cam_xmat[cam_id].reshape(3, 3)
    pc = R.T @ (np.asarray(point) - cam_pos)
    x, y, z = pc
    depth = -z
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
        if name and (name.startswith("worker") or name == "floor" or name.startswith("fl_")):
            continue

        pos = data.geom_xpos[i]
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

def is_visible(point, owner_prefix):
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
    hit_dist = mujoco.mj_ray(model, data, cam_pos, vec_norm, None, 1, -1, geomid)

    if hit_dist >= 0 and hit_dist < dist - 0.05:
        if geomid[0] != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])
            if name and name.startswith(owner_prefix):
                return True
        return False

    return True

################################################
# Sample geom surface
################################################

def sample_geom_points(gid, n_circle=12, n_height=6, n_phi=8, n_theta=12):
    center = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    pts = []

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = size[0]
        for phi in np.linspace(0, np.pi, n_phi):
            for theta in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
                local = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                pts.append(center + R @ local)

    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        rx, ry, rz = size
        for phi in np.linspace(0, np.pi, n_phi):
            for theta in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
                local = np.array([
                    rx * np.sin(phi) * np.cos(theta),
                    ry * np.sin(phi) * np.sin(theta),
                    rz * np.cos(phi)
                ])
                pts.append(center + R @ local)

    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r = size[0]
        half_length = size[1]

        for z in np.linspace(-half_length, half_length, n_height):
            for theta in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
                local = np.array([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    z
                ])
                pts.append(center + R @ local)

        for phi in np.linspace(0, np.pi / 2, n_phi // 2):
            z = half_length + r * np.cos(phi)
            rr = r * np.sin(phi)
            for theta in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
                local = np.array([
                    rr * np.cos(theta),
                    rr * np.sin(theta),
                    z
                ])
                pts.append(center + R @ local)

        for phi in np.linspace(0, np.pi / 2, n_phi // 2):
            z = -half_length - r * np.cos(phi)
            rr = r * np.sin(phi)
            for theta in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
                local = np.array([
                    rr * np.cos(theta),
                    rr * np.sin(theta),
                    z
                ])
                pts.append(center + R @ local)

    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = size
        values_x = np.linspace(-sx, sx, 8)
        values_y = np.linspace(-sy, sy, 8)
        values_z = np.linspace(-sz, sz, 8)

        for x in [-sx, sx]:
            for y in values_y:
                for z in values_z:
                    pts.append(center + R @ np.array([x, y, z]))

        for y in [-sy, sy]:
            for x in values_x:
                for z in values_z:
                    pts.append(center + R @ np.array([x, y, z]))

        for z in [-sz, sz]:
            for x in values_x:
                for y in values_y:
                    pts.append(center + R @ np.array([x, y, z]))
    else:
        pts.append(center)

    return pts

def analyze_geoms(names, owner_prefix):
    results = {}

    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue

        pixels = []
        visible = 0
        total = 0

        for p in sample_geom_points(gid):
            total += 1
            if not is_visible(p, owner_prefix):
                continue

            visible += 1
            pix = project(p)
            if pix is not None:
                pixels.append(pix)

        ratio = visible / total if total else 0.0
        results[name] = {"ratio": ratio, "pixels": pixels}

    return results

def bbox_from_pixels(pixels):
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
    if (x2 - x1) < 6 or (y2 - y1) < 6:
        return None

    return (int(x1), int(y1), int(x2), int(y2))

################################################
# Random Scene
################################################

def randomize_scene():
    workers = []
    placed_positions = []
    movable_workers = [w for w in workers_info if not w["is_sitting"]]

    active_movable = random.sample(
        movable_workers,
        random.randint(1, len(movable_workers))
    )

    for spec in workers_info:
        model.body_pos[spec["id"]] = [100.0, 100.0, -10.0]
        model.body_quat[spec["id"]] = spec["initial_quat"]

    mujoco.mj_forward(model, data)

    for spec in workers_info:
        is_active = spec in active_movable

        if spec["is_sitting"]:
            is_active = random.choice([True, False])

        if not is_active:
            continue

        bid = spec["id"]

        if spec["is_sitting"]:
            x, y = spec["initial_pos"][:2]
            model.body_pos[bid] = spec["initial_pos"]
            model.body_quat[bid] = spec["initial_quat"]
        else:
            found = False

            for _ in range(100):
                x = np.random.uniform(-4.0, 4.0)
                y = np.random.uniform(-4.0, 4.0)

                if not is_valid_position(x, y, 0.6):
                    continue
                if any(np.hypot(x - px, y - py) < 1.0 for px, py, _ in placed_positions):
                    continue

                found = True
                break

            if not found:
                continue

            model.body_pos[bid] = [x, y, spec["initial_pos"][2]]

            yaw = np.random.uniform(0, 2 * np.pi)
            yaw_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(
                yaw_quat,
                np.array([0.0, 0.0, 1.0]),
                yaw
            )

            result_quat = np.zeros(4)
            mujoco.mju_mulQuat(
                result_quat,
                yaw_quat,
                spec["initial_quat"]
            )
            model.body_quat[bid] = result_quat

        has_helmet = random.choice([True, False])
        has_vest = random.choice([True, False])
        prefix = spec["geom_prefix"]

        color = random_color(np.array([0.2, 0.25, 0.3]))

        for part in spec["body_parts"]:
            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{prefix}_{part}"
            )
            if gid >= 0:
                model.geom_rgba[gid][:3] = color

        for part, visible in [("helmet", has_helmet), ("vest", has_vest)]:
            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{prefix}_{part}"
            )
            if gid >= 0:
                model.geom_rgba[gid][3] = 1.0 if visible else 0.0

        placed_positions.append(
            (float(x), float(y), 0.45 if spec["is_sitting"] else 0.6)
        )

        workers.append({
            "name": spec["name"],
            "geom_prefix": prefix,
            "body_parts": spec["body_parts"],
            "x": float(x),
            "y": float(y),
            "helmet": has_helmet,
            "vest": has_vest,
        })

    ################################################
    # forklift position / rotation
    ################################################

    fj_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "forklift_freejoint"
    )

    if fj_id < 0:
        raise RuntimeError("forklift_freejoint not found")

    qpos_adr = model.jnt_qposadr[fj_id]

    fx = np.random.uniform(-4.0, 4.0)
    fy = np.random.uniform(-4.0, 4.0)

    data.qpos[qpos_adr:qpos_adr + 3] = [fx, fy, 0.0]

    fyaw = np.random.uniform(0, 2 * np.pi)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [
        np.cos(fyaw / 2),
        0,
        0,
        np.sin(fyaw / 2)
    ]

    ################################################
    # camera pose
    ################################################

    CAM_Z = 0.383
    target = random.choice(workers)

    for _ in range(100):
        cam_x = np.random.uniform(-4.5, 4.5)
        cam_y = np.random.uniform(-4.5, 4.5)

        if not is_valid_position(cam_x, cam_y, 0.3):
            continue

        dist = np.hypot(target["x"] - cam_x, target["y"] - cam_y)

        if 2.0 <= dist <= 6.0:
            break

    model.cam_pos[cam_id] = [cam_x, cam_y, CAM_Z]

    dx = target["x"] - cam_x
    dy = target["y"] - cam_y
    yaw = np.arctan2(dy, dx)
    yaw += np.random.uniform(np.deg2rad(-25), np.deg2rad(25))

    base_quat = np.array([0.5, 0.5, -0.5, -0.5])
    yaw_quat = np.zeros(4)

    mujoco.mju_axisAngle2Quat(
        yaw_quat,
        np.array([0, 0, 1]),
        yaw
    )

    result = np.zeros(4)
    mujoco.mju_mulQuat(result, yaw_quat, base_quat)
    model.cam_quat[cam_id] = result

    return workers

################################################
# Generate Dataset
################################################

NUM_DATA = 10000
TRAIN_RATIO = 0.8

indices = list(range(NUM_DATA))
random.seed(42)
random.shuffle(indices)
train_set = set(indices[:int(NUM_DATA * TRAIN_RATIO)])

metadata = []

for i in range(NUM_DATA):
    workers = randomize_scene()
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera="dataset_camera")
    img = renderer.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    labels = []

    ############################################
    # worker labels
    ############################################

    for worker in workers:
        prefix = worker["geom_prefix"]

        body_parts = [f"{prefix}_{part}" for part in worker["body_parts"]]
        all_parts = body_parts + [f"{prefix}_head"]

        if worker["helmet"]:
            all_parts.append(f"{prefix}_helmet")
        if worker["vest"]:
            all_parts.append(f"{prefix}_vest")

        analysis = analyze_geoms(all_parts, f"{prefix}_")

        head_name = f"{prefix}_head"
        if analysis[head_name]["ratio"] < 0.5:
            continue

        body_visible = 0
        body_total = 0

        for name in body_parts:
            result = analysis[name]
            body_visible += result["ratio"]
            body_total += 1

        body_ratio = body_visible / body_total

        if body_ratio < 0.5:
            continue

        person_pixels = []
        for name in all_parts:
            person_pixels.extend(analysis[name]["pixels"])

        person_bbox = bbox_from_pixels(person_pixels)

        if person_bbox:
            labels.append((0, person_bbox))

        if worker["helmet"]:
            helmet_name = f"{prefix}_helmet"
            helmet_bbox = bbox_from_pixels(analysis[helmet_name]["pixels"])

            if helmet_bbox:
                labels.append((1, helmet_bbox))

        if worker["vest"]:
            vest_name = f"{prefix}_vest"
            vest_bbox = bbox_from_pixels(analysis[vest_name]["pixels"])

            if vest_bbox:
                labels.append((2, vest_bbox))

    ############################################
    # forklift labels
    ############################################

    forklift_parts = [
        "fl_body",
        "fl_fork",
        "fl_mast",
        "fl_wheel_fl",
        "fl_wheel_fr",
        "fl_wheel_rl",
        "fl_wheel_rr"
    ]

    forklift_analysis = analyze_geoms(forklift_parts, "fl_")
    forklift_pixels = []
    fl_visible = 0
    fl_total = 0

    for name in forklift_parts:
        if name not in forklift_analysis:
            continue

        result = forklift_analysis[name]
        forklift_pixels.extend(result["pixels"])
        fl_visible += result["ratio"]
        fl_total += 1

    fl_ratio = fl_visible / fl_total if fl_total > 0 else 0.0
    forklift_bbox = bbox_from_pixels(forklift_pixels)

    if forklift_bbox and fl_ratio > 0.3:
        labels.append((3, forklift_bbox))

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
    # Debug bbox
    ############################################

    if DEBUG_BBOX:
        debug_img = img.copy()

        for cls, bbox in labels:
            x1, y1, x2, y2 = bbox
            color = DEBUG_COLORS.get(cls, (255, 255, 0))
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

            label_text = DEBUG_CLASS_NAMES.get(cls, str(cls))
            (tw, th), _ = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1
            )

            cv2.rectangle(
                debug_img,
                (x1, max(y1 - th - 6, 0)),
                (x1 + tw + 6, max(y1, th + 6)),
                color,
                -1
            )

            cv2.putText(
                debug_img,
                label_text,
                (x1 + 3, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(
            os.path.join(DEBUG_IMAGE_DIR, f"{name}.jpg"),
            debug_img
        )

    ############################################
    # Save YOLO
    ############################################

    with open(os.path.join(label_dir, f"{name}.txt"), "w") as f:
        for cls, bbox in labels:
            x1, y1, x2, y2 = bbox

            x1 = np.clip(x1, 0, WIDTH - 1)
            y1 = np.clip(y1, 0, HEIGHT - 1)
            x2 = np.clip(x2, 0, WIDTH - 1)
            y2 = np.clip(y2, 0, HEIGHT - 1)

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue

            cx = ((x1 + x2) / 2) / WIDTH
            cy = ((y1 + y2) / 2) / HEIGHT
            bw = (x2 - x1) / WIDTH
            bh = (y2 - y1) / HEIGHT

            if bw < 0.01 or bh < 0.01:
                continue

            f.write(
                f"{cls} {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}\n"
            )

    ############################################
    # metadata
    ############################################

    metadata.append({
        "image": name + ".jpg",
        "workers": workers,
        "camera": model.cam_pos[cam_id].tolist(),
        "camera_quat": model.cam_quat[cam_id].tolist()
    })

    if i % 100 == 0:
        print(f"{i}/{NUM_DATA}")

################################################
# Save metadata
################################################

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Dataset generation complete")