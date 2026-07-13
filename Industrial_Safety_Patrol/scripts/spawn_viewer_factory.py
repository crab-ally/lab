import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # scripts/
xml_path = BASE_DIR.parent / "scenes" / "patrol_factory.xml"

# 1. 모델 로드
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 2. 순찰 중 장애물 감지 로직 [3, 4]
def check_for_obstacles(data):
    lidar_dist = data.sensor('lidar').data[0]
    if lidar_dist >= 0 and lidar_dist < 1.0: # 1미터 이내 장애물 감지 시
        return True, lidar_dist
    return False, lidar_dist

# 3. 시뮬레이션 실행 (100Hz 루프) [5]
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [-3, -3, 0.5]
    viewer.cam.distance = 3
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -30

    print("터틀봇 실내 순찰 시뮬레이션 시작...")
    
    while viewer.is_running():
        step_start = time.time()

        state = "move"

        # 기본 주행: 전진 명령
        data.ctrl[0] = -2.0 
        data.ctrl[1] = -2.0

        # 장애물 감지 시 우회 또는 정지 로직 (Scenario 1) [4]
        is_hazard, dist = check_for_obstacles(data)
        if is_hazard:
            # 장애물 발견 시 왼쪽으로 회전하여 회피 시도
            state = "detected"
            data.ctrl[0] = -1
            data.ctrl[1] = 3
            if int(data.time * 10) % 5 == 0:
                print(f"[경고] 장애물 감지! 거리: {dist:.2f}m - 우회 경로 탐색 중...")

        if int(data.time * 10) % 5 == 0 and state == "move":
            print("전진합니다")
    
        mujoco.mj_step(model, data)
        viewer.sync()

        # 100Hz 제어 타이밍 유지
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)