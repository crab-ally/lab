import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # scripts/
xml_path = BASE_DIR.parent / "models" / "turtlebot_patrol.xml"

# 1. 모델 로드 (위의 XML 내용을 파일로 저장하거나 문자열로 로드)
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 2. 자가 진단 (Self-Diagnosis) 함수 [3], [4]
def self_diagnosis(data):
    print("시스템 자가 진단 시작...")
    # LiDAR 및 센서 데이터 연결 확인
    lidar_val = data.sensor('lidar').data
    if np.isnan(lidar_val):
        print("경고: LiDAR 센서 이상 감지!")
        return False
    
    # 배터리 가상 체크 (PRD 기준 15% 이상인지 확인) [3]
    battery_level = 100.0 
    if battery_level < 15.0:
        print("배터리 부족으로 순찰 불가")
        return False
        
    print("모든 시스템 정상. 순찰을 시작합니다.")
    return True

# 3. 메인 시뮬레이션 루프
with mujoco.viewer.launch_passive(model, data) as viewer:
    if self_diagnosis(data):
        start_time = time.time()
        
        # 100Hz 제어 루프 유지 [1]
        while viewer.is_running():
            step_start = time.time()

            # 로봇 제어: 전진 (왼쪽/오른쪽 바퀴에 속도 인가)
            data.ctrl[0] = -1.0  # 왼쪽 바퀴 속도
            data.ctrl[1] = -1.0  # 오른쪽 바퀴 속도

            # 물리 엔진 스텝 업데이트 [3]
            mujoco.mj_step(model, data)

            # 센서 데이터 로깅 (1초마다 출력) [5]
            if int(data.time) > int(data.time - 0.01) and int(data.time) % 1 == 0:
                dist = data.sensor('lidar').data
                print(f"Time: {data.time:.2f}s | LiDAR 거리 감지: {dist}")

            # 뷰어 동기화
            viewer.sync()

            # 100Hz 타이밍 제어
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)