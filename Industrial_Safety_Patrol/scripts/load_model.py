import mujoco
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
xml_path = BASE_DIR.parent / "models" / "turtlebot_patrol.xml"

# 모델 로드
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print("=== Industrial Safety Patrol Model Loaded ===")
print(f"Number of bodies: {model.nbody}")
print(f"Number of joints: {model.njnt}")
print(f"Number of geoms: {model.ngeom}")
print(f"Number of actuators: {model.nu}")
print(f"Number of sensors: {model.nsensor}")
print()

print("=== Actuator List ===")
for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )

    trn_id = model.actuator_trnid[actuator_id]
    ctrl_range = model.actuator_ctrlrange[actuator_id]
    ctrl_limited = model.actuator_ctrllimited[actuator_id]

    joint_id = trn_id[0]
    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    print(
        f"[{actuator_id:03d}] "
        f"name={name}, "
        f"joint={joint_name}, "
        f"ctrl_limited={ctrl_limited}, "
        f"ctrl_range={ctrl_range}"
    )
print()

print("=== Sensor List ===")
for sensor_id in range(model.nsensor):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        sensor_id,
    )

    data_addr = model.sensor_adr[sensor_id]
    data_dim = model.sensor_dim[sensor_id]

    print(
        f"[{sensor_id:03d}] "
        f"name={name}, "
        f"data_addr={data_addr}, "
        f"dim={data_dim}"
    )