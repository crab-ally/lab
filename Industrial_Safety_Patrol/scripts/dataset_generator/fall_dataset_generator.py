#!/usr/bin/env python3
"""
Fall Detection 데이터셋 생성기

쓰러진 작업자(class 0: fallen)와 서있는 작업자(class 1: standing)의
Bounding Box 어노테이션을 포함한 YOLO 형식 데이터셋을 생성합니다.

클래스 정의:
    0: fallen  (쓰러진 작업자)
    1: standing (서있는 작업자)

출력 구조:
    fall_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── metadata.json
"""

import mujoco
import numpy as np
import cv2
import os
import json
import random

XML_PATH = "/workspace/worlds/fall_dataset_world.xml"

DATASET = "/workspace/datasets/fall_dataset"

TRAIN_IMAGE = os.path.join(DATASET, "images/train")
VAL_IMAGE   = os.path.join(DATASET, "images/val")
TRAIN_LABEL = os.path.join(DATASET, "labels/train")
VAL_LABEL   = os.path.join(DATASET, "labels/val")
META_PATH   = os.path.join(DATASET, "metadata.json")

for path in [TRAIN_IMAGE, VAL_IMAGE, TRAIN_LABEL, VAL_LABEL]:
    os.makedirs(path, exist_ok=True)

################################################
# Load MuJoCo
################################################

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

cam_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "dataset_camera"
)

################################################
# Worker IDs 수집
################################################

NUM_FALLEN   = 5
NUM_STANDING = 3

fallen_ids   = []
standing_ids = []

for i in range(NUM_FALLEN):
    bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"fallen_worker_{i}"
    )
    fallen_ids.append(bid)

for i in range(NUM_STANDING):
    bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"standing_worker_{i}"
    )
    standing_ids.append(bid)

################################################
# Renderer
################################################

WIDTH  = 640
HEIGHT = 480

renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

################################################
# Helper: 색상 랜덤화
################################################

def random_color(base, delta=0.15):
    result = []
    for c in base:
        value = c + np.random.uniform(-delta, delta)
        result.append(float(np.clip(value, 0, 1)))
    return result

################################################
# Helper: 3D → 픽셀 투영
################################################

def project(point):
    cam_pos = data.cam_xpos[cam_id]
    R       = data.cam_xmat[cam_id].reshape(3, 3)
    pc      = R.T @ (np.asarray(point) - cam_pos)
    x, y, z = pc
    depth = -z  # 카메라는 -Z 방향을 바라봄
    if depth <= 1e-6:
        return None
    fovy = model.cam_fovy[cam_id]
    f    = HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))
    u = WIDTH  / 2 + f * x / depth
    v = HEIGHT / 2 - f * y / depth
    return (u, v)

################################################
# Helper: 충돌 회피 위치 유효성 검사
################################################

def is_valid_position(x, y, radius=0.5):
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and (
            name.startswith("fallen_worker")
            or name.startswith("standing_worker")
            or name == "floor"
        ):
            continue
        pos   = model.geom_pos[i]
        size  = model.geom_size[i]
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
# Helper: 가시성 판단 (레이 캐스팅)
################################################

def is_visible(point, prefix):
    pix = project(point)
    if pix is None:
        return False
    u, v = pix
    if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
        return False
    cam_pos  = data.cam_xpos[cam_id]
    vec      = point - cam_pos
    dist     = np.linalg.norm(vec)
    if dist < 1e-6:
        return True
    vec_norm = vec / dist
    geomid   = np.array([-1], dtype=np.int32)
    hit_dist = mujoco.mj_ray(model, data, cam_pos, vec_norm, None, 1, -1, geomid)
    if hit_dist >= 0 and hit_dist < dist - 0.05:
        if geomid[0] != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])
            if name and name.startswith(prefix):
                return True
        return False
    return True

################################################
# Helper: geom 표면 점 샘플링
################################################

def sample_geom_points(gid, n_circle=24, n_height=8):
    center = data.geom_xpos[gid]
    R      = data.geom_xmat[gid].reshape(3, 3)
    size   = model.geom_size[gid]
    gtype  = model.geom_type[gid]
    pts    = []

    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        for theta in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
            for z in np.linspace(-h, h, n_height):
                local = np.array([r * np.cos(theta), r * np.sin(theta), z])
                pts.append(center + R @ local)

    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = size[0]
        for phi in np.linspace(0, np.pi, 10):
            for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                local = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                pts.append(center + R @ local)

    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = size
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):
                    pts.append(center + R @ np.array([dx, dy, dz]))
    else:
        pts.append(center)

    return pts

################################################
# Helper: 가시 비율 계산
################################################

def visible_ratio(geom_names, prefix):
    visible = 0
    total   = 0
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        for p in sample_geom_points(gid):
            total += 1
            if is_visible(p, prefix):
                visible += 1
    return (visible / total) if total > 0 else 0.0

################################################
# Helper: geom 목록에서 Bounding Box 계산
################################################

def bbox_from_geoms(geom_names, prefix):
    pixels = []
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        for p in sample_geom_points(gid):
            if not is_visible(p, prefix):
                continue
            pix = project(p)
            if pix is not None:
                pixels.append(pix)
    if len(pixels) < 20:
        return None
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x1 = int(np.clip(min(xs), 0, WIDTH  - 1))
    y1 = int(np.clip(min(ys), 0, HEIGHT - 1))
    x2 = int(np.clip(max(xs), 0, WIDTH  - 1))
    y2 = int(np.clip(max(ys), 0, HEIGHT - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 6 or (y2 - y1) < 6:
        return None
    return (x1, y1, x2, y2)

################################################
# Helper: 쓰러진 자세 쿼터니언 생성
# - 몸체를 바닥에 눕히기 위해 X축 또는 Y축으로 90도 회전 후
#   수평 방향(Yaw)은 랜덤으로 변경
################################################

def make_fallen_quat(yaw):
    """
    1) X축으로 90도 회전 → 실린더가 바닥에 눕힘
    2) Z축으로 yaw 회전 → 누운 방향 랜덤화
    """
    # X축 90도 회전: 실린더의 길이 방향이 수평이 됨
    tilt_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(tilt_quat, np.array([1.0, 0.0, 0.0]), np.pi / 2)

    # Z축 yaw 회전
    yaw_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), yaw)

    # 합성 (먼저 yaw, 그다음 tilt)
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, yaw_quat, tilt_quat)
    return result

################################################
# 씬 랜덤화
################################################

def randomize_scene():
    """
    씬을 랜덤으로 구성하고 작업자 메타 정보를 반환합니다.

    Returns:
        list[dict]: 작업자 정보 목록
            - id: 인덱스 번호
            - type: "fallen" 또는 "standing"
            - x, y: 위치
    """
    workers = []

    # ─── 쓰러진 작업자 수 선택 (1~NUM_FALLEN) ─────────────────────
    n_fallen = random.randint(1, NUM_FALLEN)

    for i, bid in enumerate(fallen_ids):
        if i >= n_fallen:
            # 화면 밖으로 숨김
            model.body_pos[bid] = [100.0, 100.0, -10.0]
            continue

        # 위치 무작위 선택 (충돌 회피)
        for _ in range(100):
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            if is_valid_position(x, y, 0.7):
                break

        # 쓰러진 자세: 몸체를 바닥에 눕힘
        yaw = np.random.uniform(0, 2 * np.pi)
        fallen_quat = make_fallen_quat(yaw)

        # 쓰러진 몸체의 지면 높이 조정
        # 실린더 반지름(0.20) 만큼 위로 올려서 바닥에 닿게
        model.body_pos[bid]  = [x, y, 0.20]
        model.body_quat[bid] = fallen_quat

        # 색상 랜덤화
        color = random_color([0.2, 0.24, 0.28])
        for part in ["torso", "pelvis", "luleg", "llleg", "ruleg", "rlleg", "larm", "rarm"]:
            body_gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"fw{i}_{part}"
            )
            if body_gid >= 0:
                model.geom_rgba[body_gid][:3] = color

        workers.append({
            "id":   i,
            "type": "fallen",
            "x":    float(x),
            "y":    float(y),
        })

    # ─── 서있는 작업자 수 선택 (0~NUM_STANDING) ───────────────────
    n_standing = random.randint(0, NUM_STANDING)

    for i, bid in enumerate(standing_ids):
        if i >= n_standing:
            model.body_pos[bid] = [100.0, 100.0, -10.0]
            continue

        for _ in range(100):
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            if is_valid_position(x, y, 0.5):
                break

        yaw = np.random.uniform(0, 2 * np.pi)
        model.body_pos[bid]  = [x, y, 0.0]
        model.body_quat[bid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        # 색상 랜덤화
        color = random_color([0.2, 0.24, 0.28])
        for part in ["torso", "pelvis", "luleg", "llleg", "ruleg", "rlleg", "larm", "rarm"]:
            body_gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"sw{i}_{part}"
            )
            if body_gid >= 0:
                model.geom_rgba[body_gid][:3] = color

        workers.append({
            "id":   i,
            "type": "standing",
            "x":    float(x),
            "y":    float(y),
        })

    # ─── 카메라 위치 / 방향 랜덤화 ────────────────────────────────
    CAM_Z = 2.0  # 감시 카메라 높이 (쓰러진 작업자를 위에서 내려다봄)

    # 바라볼 쓰러진 작업자 1명 선택
    fallen_workers = [w for w in workers if w["type"] == "fallen"]
    target = random.choice(fallen_workers)

    for _ in range(100):
        cam_x = np.random.uniform(-4.0, 4.0)
        cam_y = np.random.uniform(-4.0, 4.0)
        if not is_valid_position(cam_x, cam_y, 0.3):
            continue
        dist = np.hypot(target["x"] - cam_x, target["y"] - cam_y)
        if 2.0 <= dist <= 7.0:
            break

    model.cam_pos[cam_id] = [cam_x, cam_y, CAM_Z]

    dx  = target["x"] - cam_x
    dy  = target["y"] - cam_y
    yaw = np.arctan2(dy, dx) + np.random.uniform(np.deg2rad(-20), np.deg2rad(20))

    # 아래를 바라보는 기본 쿼터니언 (카메라 로컬 Z = 월드 -Z)
    base_quat = np.array([0.5, 0.5, -0.5, -0.5])
    yaw_quat  = np.zeros(4)
    mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0, 0, 1]), yaw)
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, yaw_quat, base_quat)
    model.cam_quat[cam_id] = result

    return workers

################################################
# 데이터셋 생성
################################################

NUM_DATA    = 8000
TRAIN_RATIO = 0.8

# 클래스 정의
CLASS_FALLEN   = 0  # 쓰러진 작업자
CLASS_STANDING = 1  # 서있는 작업자

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

    # ─── 쓰러진 작업자 어노테이션 ────────────────────────────────
    for w in workers:
        if w["type"] != "fallen":
            continue

        idx    = w["id"]
        prefix = f"fw{idx}_"

        body_parts = [
            f"{prefix}torso", f"{prefix}pelvis", f"{prefix}luleg", f"{prefix}llleg",
            f"{prefix}ruleg", f"{prefix}rlleg", f"{prefix}larm", f"{prefix}rarm"
        ]

        body_ratio = visible_ratio(body_parts, prefix)
        if body_ratio < 0.3:
            continue

        geom_names = body_parts + [f"{prefix}head"]
        bbox = bbox_from_geoms(geom_names, prefix)
        if bbox:
            labels.append((CLASS_FALLEN, bbox))

    # ─── 서있는 작업자 어노테이션 ────────────────────────────────
    for w in workers:
        if w["type"] != "standing":
            continue

        idx    = w["id"]
        prefix = f"sw{idx}_"

        head_ratio = visible_ratio([f"{prefix}head"], prefix)
        if head_ratio < 0.5:
            continue

        body_parts = [
            f"{prefix}torso", f"{prefix}pelvis", f"{prefix}luleg", f"{prefix}llleg",
            f"{prefix}ruleg", f"{prefix}rlleg", f"{prefix}larm", f"{prefix}rarm"
        ]

        body_ratio = visible_ratio(body_parts, prefix)
        if body_ratio < 0.5:
            continue

        geom_names = body_parts + [f"{prefix}head"]
        bbox = bbox_from_geoms(geom_names, prefix)
        if bbox:
            labels.append((CLASS_STANDING, bbox))

    # ─── 저장 경로 결정 ──────────────────────────────────────────
    name = f"{i:06d}"
    if i in train_set:
        image_dir = TRAIN_IMAGE
        label_dir = TRAIN_LABEL
    else:
        image_dir = VAL_IMAGE
        label_dir = VAL_LABEL

    # ─── 이미지 저장 ─────────────────────────────────────────────
    cv2.imwrite(os.path.join(image_dir, f"{name}.jpg"), img)

    # ─── YOLO 라벨 저장 ──────────────────────────────────────────
    with open(os.path.join(label_dir, f"{name}.txt"), "w") as f:
        for cls, bbox in labels:
            x1, y1, x2, y2 = bbox

            x1 = int(np.clip(x1, 0, WIDTH  - 1))
            y1 = int(np.clip(y1, 0, HEIGHT - 1))
            x2 = int(np.clip(x2, 0, WIDTH  - 1))
            y2 = int(np.clip(y2, 0, HEIGHT - 1))

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

            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # ─── 메타데이터 ───────────────────────────────────────────────
    metadata.append({
        "image":       name + ".jpg",
        "workers":     workers,
        "camera":      model.cam_pos[cam_id].tolist(),
        "camera_quat": model.cam_quat[cam_id].tolist(),
    })

    if i % 100 == 0:
        print(f"[{i}/{NUM_DATA}] labels={len(labels)}")

################################################
# 메타데이터 저장
################################################

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Fall dataset generation complete!")
print(f"  Train: {len(train_set)} images")
print(f"  Val  : {NUM_DATA - len(train_set)} images")
