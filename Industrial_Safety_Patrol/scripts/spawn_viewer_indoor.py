import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # scripts/
xml_path = BASE_DIR.parent / "scenes" / "patrol_indoor.xml"

# 1. 모델 로드
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 2. 시뮬레이션 실행 (100Hz 루프)
count = 0 
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [-3, -3, 0.5]
    viewer.cam.distance = 3
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -30

    print("터틀봇 실내 순찰 시뮬레이션 시작...")
    
    start_time = time.time()
    while viewer.is_running():
        step_start = time.time()

        # 기본 주행: 전진 명령
        data.ctrl[0] = -2.0 
        data.ctrl[1] = -2.0
        if count == 0:
                print("전진")
                count += 1

        if time.time() - start_time > 20:
            data.ctrl[0] = -2.0
            data.ctrl[1] = -2.0
            if count == 4:
                print("전진")
                count += 1
        elif time.time() - start_time > 15:
            data.ctrl[0] = -1.0
            data.ctrl[1] = -3.0
            if count == 3:
                print("좌회전")
                count += 1
        elif time.time() - start_time > 10:
            data.ctrl[0] = -2.0
            data.ctrl[1] = -2.0
            if count == 2:
                print("전진")
                count += 1
        elif time.time() - start_time > 5:
            data.ctrl[0] = -3.0
            data.ctrl[1] = -1.0
            if count == 1:
                print("우회전")
                count += 1

        mujoco.mj_step(model, data)
        viewer.sync()

        # 100Hz 제어 타이밍 유지
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)