import mujoco
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
xml_path = BASE_DIR.parent / "scenes" / "patrol_env.xml"

# 1. 모델 로드
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 2. LiDAR 센서 ID 확인
lidar_id = model.sensor('lidar').id

print("LiDAR sensor ID:", lidar_id)
print("Sensor type:", model.sensor('lidar').type)

# 3. 초기 센서값
print("Initial LiDAR:", data.sensor('lidar').data)


# 4. 시뮬레이션 실행
for i in range(1000):

    # 전진
    data.ctrl[0] = -2.0
    data.ctrl[1] = -2.0

    # 물리 계산
    mujoco.mj_step(model, data)

    # 10 step마다 출력
    if i % 10 == 0:
        lidar_data = data.sensor('lidar').data.copy()

        print(
            f"time={data.time:.3f}s, "
            f"LiDAR={lidar_data}"
        )

    time.sleep(model.opt.timestep)


print("Simulation finished")