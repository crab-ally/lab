# 에러 정정표 (Troubleshooting Log)

AI Safety Patrol Robot 프로젝트 진행 중 발생한 오류와 해결 기록

---

# 1. NumPy 배열 포맷팅 오류

## 파일

`scripts/spawn_world.py`

## 에러 코드

```text
TypeError: unsupported format string passed to numpy.ndarray.__format__
```

## 기존 코드

```python
print(f"[경고] 장애물 감지! 거리: {dist:.2f}m - 우회 경로 탐색 중...")
```

## 원인

`data.sensor(...).data`는 단일 숫자가 아니라 **NumPy 배열(ndarray)** 형태로 반환된다.

하지만 `.2f`는 Python 실수(float) 출력 형식이며 배열에는 적용할 수 없다.

## 해결

```python
lidar_dist = data.sensor('lidar').data
                ↓
lidar_dist = data.sensor('lidar').data[0]
```

---

# 2. LiDAR 장애물 오검출

## 파일

`scripts/spawn_world.py`

## 문제

장애물이 없음에도 장애물 감지 메시지가 출력됨.

## 기존 코드

```python
if lidar_dist < 1.0:
```

## 원인

MuJoCo LiDAR는 장애물을 찾지 못하면 `-1`을 반환한다.

## 해결

```python
if lidar_dist < 1.0:
        ↓
if lidar_dist >= 0 and lidar_dist < 1.0:
```

---

# 3. LiDAR 방향 오류

## 파일

`models/turtlebot_patrol.xml`

## 문제

시뮬레이션 시작 직후 장애물이 감지됨.

## 원인

LiDAR 센서 방향(-Y)이 의도한 방향(+Y)과 반대로 설정됨.
따라서 뒤쪽 충전 스테이션을 장애물로 감지함.

## 해결

```xml
<site name="lidar_site"
      pos="0.07 0 0"
      quat="0.707 0 -0.707 0"
      size="0.005"/>
        ↓
<site name="lidar_site"
      pos="0.07 0 0"
      quat="0.707 0 0.707 0"
      size="0.005"/>
```

---

# 4. 충전 스테이션 탈출 시 회전 문제

## 파일

`scenes/patrol_env.xml`

## 문제

로봇이 충전 스테이션에서 내려올 때 회전함.

## 원인

로봇이 충전 스테이션에서 내려올 때 본체가 기울어져 바닥을 장애물로 감지함.

## 임시 해결

충전 스테이션 바닥 제거.

---

# 5. Moving Body 질량/관성 오류

## 파일

`worlds/industrial_factory.xml`

## 에러 코드

```text
ValueError:
mass and inertia of moving bodies must be larger than mjMINVAL

Element name 'forklift'
```

## 원인

움직이는 Body(`forklift`)의 질량 또는 관성이 0으로 계산됨.

MuJoCo에서는 움직이는 객체는 반드시:

```text
mass > 0
inertia > 0
```

조건을 만족해야 한다.

## 발생 원인

`freejoint` 추가 후 자유롭게 움직이는 객체로 되었지만 물리 파라미터가 부족함.

## 임시 해결

freejoint 제거. 이후 단계적으로 추가 예정.

---

# 6. Docker pip 설치 중 Sympy 충돌

## 파일

`Dockerfile`

## 문제

Docker Image Build 과정에서 `pip install` 단계 실패.

## 에러 원인

`osrf/ros:humble-desktop-full` 이미지 내부에는 ROS2 패키지 관리를 위해 `apt`로 설치된 Python 패키지가 존재한다.

그중:

```text
sympy 1.9
```

가 이미 설치되어 있다.

이후:

```text
pip install torch
        ↓
torch dependency 확인
        ↓
sympy 최신 버전 설치 시도
        ↓
apt package와 pip package 충돌
```

발생.

## 해결

Python Virtual Environment 생성.

추가:

```dockerfile
RUN python3 -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
```

---

# 7. NumPy 2.x / ROS2 cv_bridge 호환 오류

## 파일

`Dockerfile`

## 에러 코드

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.

AttributeError: _ARRAY_API not found
```

## 원인

ROS2 Humble의 `cv_bridge`는 NumPy 1.x 기준으로 컴파일된 Binary Package이다.

> Binary Package: 이미 컴파일되어 바로 사용할 수 있는 패키지

하지만 Docker 환경에서:

```text
pip install numpy
        ↓
numpy 2.2.6 설치
```

됨.

결과: ABI 호환 오류

> ABI (Application Binary Interface): 컴파일된 프로그램과 라이브러리가 데이터를 주고받는 규칙

## 해결

기존:

```dockerfile
numpy
  ↓
numpy==1.26.4
```

---

# 8. MuJoCo Renderer OpenGL Context 오류

## 파일

`scripts/mujoco_ros2_bridge.py`

## 에러 코드

```text
Renderer initialization failed,
camera data will not be published

gladLoadGL error
```

## 원인

기존 구조에서는 ROS2 Node 생성 시점에 Renderer를 생성했다.

기존:

```python
class MujocoRosBridge(Node):

    def __init__(self):
        self.renderer = mujoco.Renderer(...)
```

실행 흐름:

```text
ROS2 Node 생성
        ↓
Renderer 생성
        ↓
OpenGL Context 없음
        ↓
Renderer 초기화 실패
```

## 이유

MuJoCo Renderer는 OpenGL Context가 필요하다.

> OpenGL Context: GPU가 그래픽 작업을 수행하기 위한 실행 환경.
> Renderer = 그림을 그리는 사람, OpenGL Context = 그림을 그릴 작업 공간.

> Renderer: 센서 이미지 또는 화면을 생성하는 모듈.
> Camera Sensor → MuJoCo Renderer → Image Message → ROS2 Topic /camera/image_raw.

하지만 OpenGL Context는 Viewer 실행 과정에서 생성된다.

즉:

```text
Renderer 생성

필요:
OpenGL Context

현재 상태:
없음

결과:
실패
```

---

## 해결

### 변경 전

```python
class MujocoRosBridge(Node):

    def __init__(self):
        self.renderer = mujoco.Renderer(
            self.model,
            480,
            640
        )
```

### 변경 후

ROS2 Node 생성 시 Renderer 생성하지 않음.

```python
class MujocoRosBridge(Node):

    def __init__(self):
        self.renderer = None
```

Viewer 실행 이후 Renderer 생성:

```python
with mujoco.viewer.launch_passive(model, data) as viewer:

    node = MujocoRosBridge(model, data)

    node.renderer = mujoco.Renderer(
        model,
        480,
        640
    )
```

---

# 9. Docker GLFW 초기화 실패

## 파일

`docker-compose.yml`

## 에러 코드

```text
ERROR: could not initialize GLFW
```

> GLFW: OpenGL 기반 프로그램에서 Window 생성, Keyboard 및 Mouse 입력 처리, OpenGL Context 생성을 담당하는 라이브러리.
> MuJoCo Viewer가 사용한다.

## 원인

Docker Container는 기본적으로 Host PC의 GUI 환경에 접근할 수 없다.

구조:

```text
Host PC
 └─ Display Server
        |
       X11
        |
Container
 └─ MuJoCo Viewer
```

Container에서 Host Display 접근 설정 필요.

> X11: Linux 계열 운영체제에서 화면 출력과 GUI 입력을 관리하는 시스템. Docker GUI 프로그램 실행 시 Host와 Container 사이 연결이 필요하다.

## 해결

기존:

```yaml
DISPLAY=${DISPLAY:-:0}
```

변경:

```yaml
DISPLAY=host.docker.internal:0.0
QT_X11_NO_MITSHM=1
MUJOCO_GL=glfw
```

제거:

```yaml
/tmp/.X11-unix:/tmp/.X11-unix:rw
```

---

# 10. Docker `/dev/dri` GPU Device 오류

## 파일

`docker-compose.yml`

## 에러 코드

```text
error gathering device information while adding custom device "/dev/dri":
not a device node
```

> Device Node: Linux에서 하드웨어 장치를 파일처럼 접근하기 위한 인터페이스

## 원인

기존 설정:

```yaml
devices:
  - /dev/dri:/dev/dri
```

은 Linux 환경에서 GPU 장치를 Container에 전달하기 위한 설정이다.

하지만 현재 환경:

```text
Windows + Docker Desktop
```

에서는 Linux GPU Device Node인:

```text
/dev/dri
```

가 존재하지 않는다.

따라서 Docker가 해당 장치를 찾지 못하고 오류 발생.

## 해결

삭제:

```yaml
devices:
  - /dev/dri:/dev/dri
```

---

# 11. OpenGL Software Rendering 문제

## 파일

`docker-compose.yml`

## 에러 코드

```text
OpenGL renderer string: softpipe

OpenGL version string: 1.4
```

## 문제

MuJoCo 실행 시 GPU 가속이 아닌 CPU 기반 Software Rendering 사용.

## 원인

Docker Container가 Host GPU/OpenGL 환경을 제대로 전달받지 못함.

결과:

```text
Host GPU
   ↓
Container 접근 실패
   ↓
Mesa Software Renderer 사용
```


> Mesa: Linux에서 OpenGL 기능을 제공하는 오픈소스 그래픽 라이브러리.
> GPU 연결이 실패하면 Software Renderer로 동작한다.

> OpenGL Rendering: 3D 그래픽을 계산하여 화면을 생성하는 과정  
> Hardware Rendering - GPU 사용 [빠름]  
> Software Rendering - CPU 사용 [느림]

## 해결

```yaml
DISPLAY=${DISPLAY:-:0}
        ↓
DISPLAY=host.docker.internal:0.0
```

---

# 12. Docker Compose Network 설정 충돌

## 파일

`docker-compose.yml`

## 에러 코드

```text
service robot_sim declares mutually exclusive
network_mode and networks
```

## 원인

Docker Compose에서는:

```yaml
network_mode:
```

와

```yaml
networks:
```

를 동시에 사용할 수 없다.

> network_mode: Container가 네트워크를 사용하는 방식을 지정  
> networks: Docker 내부 가상 네트워크를 생성하고 Container끼리 연결

## 해결

`network_mode` 제거.

---

# 13. MuJoCo Viewer Segmentation Fault

## 파일

`scripts/mujoco_ros2_bridge.py`

## 에러 코드

```text
Segmentation fault

X_GLXMakeCurrent

BadAccess
```

> Segmentation Fault: 프로그램이 접근하면 안 되는 메모리에 접근했을 때 발생하는 오류

> GLX: Linux 환경에서 OpenGL과 X Window System을 연결하는 인터페이스.
> X_GLXMakeCurrent: OpenGL Context 연결 실패.

## 문제

MuJoCo Viewer 실행 중 비정상 종료 발생.

## 원인

OpenGL Context를 서로 다른 Thread에서 접근함.

기존 구조:

```text
Main Thread
 └─ MuJoCo Viewer
       |
       |
       OpenGL Context 생성

ROS Thread
 └─ Renderer 생성
       |
       |
       OpenGL Context 접근
```

OpenGL Context는 기본적으로 생성된 Thread에서만 안전하게 사용 가능하다.

## 해결

Thread 구조 변경.

수정 후:

```text
Main Thread
 ├─ MuJoCo Viewer 실행
 ├─ OpenGL Context 생성
 └─ Renderer 생성

ROS Thread
 └─ ROS2 spin()
```

즉:

```text
Viewer
  +
Renderer
  ↓
같은 Thread에서 실행

ROS 통신
   ↓
별도 Thread 처리
```

---

# 14. MuJoCo 로봇 이동 방향 반대 문제

파일: `models/turtlebot_patrol.xml`

문제: 로봇에게 전진 명령을 줬는데 실제 시뮬레이션에서는 후진함

원인: MJCF에서 바퀴 회전축 방향(axis)이 실제 이동 방향과 반대로 설정됨

해결
```xml
<!-- 왼쪽 바퀴도 동일하게 변경 -->
<joint name="drive_right" axis="0 0 1"/>
                ↓
<joint name="drive_right" axis="0 0 -1">
```

---

# 15. Auto Explore 종료 후 로봇 자동 재시작 문제

파일: `scripts/auto_explore_for_mapping.py`

문제: 자동 탐색 중 종료했는데 MuJoCo Viewer만 다시 실행해도 로봇이 이전 경로로 자동 이동. 심지어 중단했던 위치로 이동 시도.

원인: Python ROS2 노드 프로세스가 백그라운드에 남아있음.

즉,

```
MuJoCo 실행
      ↑
      |
남아있는 auto_explore process
      |
      ↓
/cmd_vel 재발행
```

해결

1. 정상 종료 시 속도 명령 초기화

```py
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
        ↓
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    node.get_logger().info(
        "Keyboard interrupt received. Stopping robot."
    )
finally:
    node.publish_cmd(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()
```

2. 종료 전에 Timer 제거

```py
if self.timer:
    self.timer.cancel()
```

3. 종료 시 Twist 0 여러 번 Publish

```py
finally:

    for _ in range(5):
        node.publish_cmd(0.0,0.0)
        time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()
```

---

# 16. SLAM Toolbox 지도 저장 실패

파일: `scripts/save_factory_map.py`

에러 코드:
```text
[map_saver]: Failed to spin map subscription

Map save failed.
Check that /map is being published.
```

원인: SLAM Toolbox가 실행되지 않은 상태에서 map 저장 실행. 즉, /map 토픽이 발행되지 않음.

해결
```
robot_sim 실행
        ↓
slam_toolbox 실행
        ↓
auto_explore 실행
        ↓
map_saver 실행
```

---

# 17. ROS2 Diagnostic Updater 라이브러리 오류

## 명령어

`docker compose --profile nav2 up nav2 rviz`

## 에러 코드

```bash
Failed to load library:
libdiagnostic_updater.so:
cannot open shared object file:
No such file or directory
```

## 원인

Nav2의 nav2_lifecycle_manager는 libdiagnostic_updater.so 라이브러리에 의존.

그런데 현재 Docker 이미지에는 ros-humble-diagnostic-updater 패키지가 설치되어 있지 않음.

## 해결

`ros-humble-diagnostic-updater` 패키지 설치.

---

# 18. Nav2 Planner Plugin 이름 오류

## 파일

`config/nav2_params.yaml`

## 에러코드

```
[FATAL] [planner_server]:
Failed to create global planner.

Exception:
According to the loaded plugin descriptions the class
nav2_navfn_planner::NavfnPlanner
with base class type nav2_core::GlobalPlanner does not exist.

Declared types are

nav2_navfn_planner/NavfnPlanner
nav2_smac_planner/SmacPlanner2D
nav2_smac_planner/SmacPlannerHybrid
nav2_smac_planner/SmacPlannerLattice
nav2_theta_star_planner/ThetaStarPlanner
```

## 원인

현재 Nav2 Humble에서는 plugin 이름이 변경되었습니다.


## 해결

```yaml
plugin: "nav2_navfn_planner::NavfnPlanner"
                 ↓
plugin: "nav2_navfn_planner/NavfnPlanner"
```

---

# 19. Rviz 로봇 위치 확인 불가

## 문제

rviz에서 display - robotmodel - description topic /robot_description 설정했는데도 지도에 로봇이 표시가 안 됨

## 원인

`robot_state_publisher` 노드 없음

## 해결

`urdf/turtlebot_patrol.urdf` 파일 생성 및 연결

---

# 20. Rviz TF 에러

## 파일

`scripts/mujoco_ros2_bridge.py`

## 에러코드

```text
RobotModel
    |
Status : Error
    |
base_footprint No transform from [base_footprint] to [map]
```

## 원인

base_link가 부모를 두 개 가져 TF 트리가 충돌함

```
odom → base_link
base_footprint → base_link
```

## 해결

```python
msg.child_frame_id = 'base_link'
odom_to_base.child_frame_id = 'base_link'
                ↓
msg.child_frame_id = 'base_footprint'
odom_to_base.child_frame_id = 'base_footprint'
```

---

# 21. waypoints 주행 unpack 오류

## 파일

`scripts/auto_explore_for_mapping.py`

## 에러코드

```text
[ERROR] [1784773009.828672112] [auto_explore_for_mapping]: too many values to unpack (expected 3)
```

## 원인

`(1.0, -1,0, 2,0)` 처럼 콤마를 잘못 입력하여 전달되는 값이 5개가 됨

## 해결

`,` 수정

---

# 22. slam 장애물 인식 오류

## 파일

`scripts/mujoco_ros2_bridge.py`

## 문제

벽과 장애물을 제대로 인식하지 못함

## 원인

```py
msg.ranges = [
    float(r) if r > 0 else float("inf")
    for r in sensor_data
]
```

라이다 스펙 상 관측값이 3.5m가 측정되면 장애물이 없는 것인데,
위 코드로는 3.5m 부근에 장애물이 있다고 해석됨.

## 해결

```py
clean_ranges = []
for r in sensor_data:
    val = float(r)
    if val >= msg.range_max - 0.05 or val < msg.range_min:
        clean_ranges.append(float("inf"))
    else:
        clean_ranges.append(val)

msg.ranges = clean_ranges
```

라이다 스펙 상 최대 거리는 inf 처리하여 장애물 없는 것으로 인식

---

# 23. Nav2 odom 변환(TF) 타임아웃 및 Odom 발행 주기 저하 오류

## 파일

`scripts/mujoco_ros2_bridge.py`

## 문제

Nav2 및 RViz2 실행 시 odom -> base_footprint TF를 찾지 못하고 타임아웃 에러 발생

```
Invalid frame ID "odom" passed to canTransform argument target_frame - frame does not exist
```

## 원인

main() 루프 내에서 매 물리 step(5ms)마다 node.renderer.render()를 통한 카메라 오프스크린 렌더링을 무겁게 실행함.
이로 인해 시뮬레이션 루프에 심각한 병목(Bottleneck)이 발생하여, 설정된 Odom/TF 발행 주기가 원래 목표치(20Hz)에 크게 못 미치는 ~1.14 Hz까지 저하됨.

Nav2는 최소 10~15Hz 이상의 TF 갱신을 요구하므로 TF 데이터를 수신하지 못해 대기 상태에 빠짐.

```py
# 기존 코드 (매 step마다 카메라 렌더링 호출되어 병목 발생)
if node.renderer is not None:
    node.renderer.update_scene(data, camera="patrol_camera")
    pixels = node.renderer.render()
    with node.camera_lock:
        node.camera_image = pixels
    node.publish_camera(stamp)
```

## 해결

```py
# 해결 코드 (카메라 렌더링 주기를 0.1초 간격으로 제한)
if data.time >= last_camera_time + 0.1:
    if node.renderer is not None:
        node.renderer.update_scene(data, camera="patrol_camera")
        pixels = node.renderer.render()
        with node.camera_lock:
            node.camera_image = pixels
        node.publish_camera(stamp)
    last_camera_time = data.time
```

---

# 24. ROS2 Python 패키지 Ultralytics import 오류

## 파일

`Dockerfile`

## 에러코드

perception_node 실행 시 Ultralytics가 설치되어 있음에도 다음 오류 발생.

```txt
ModuleNotFoundError: No module named 'ultralytics'
```

Docker 컨테이너 내부에서 직접 Python을 실행하면 Ultralytics가 정상적으로 import됨.
하지만 ros2 launch로 실행하면 perception_node의 실행 스크립트가 system Python을 사용함.

```
#!/usr/bin/python3
```

이로 인해 /usr/bin/python3 환경에서 ultralytics를 찾지 못함.

## 원인

Dockerfile에서 ROS2의 colcon은 apt를 통해 system Python 환경에 설치되어 있었음.
반면 AI 관련 패키지는 /opt/venv에 설치되어 있었음

```txt
/usr/bin/python3
└── ROS2 / colcon
    └── ultralytics 없음

/opt/venv/bin/python3
└── ultralytics
└── DeepSORT
└── MuJoCo
└── NumPy
```

따라서 /usr/bin/colcon으로 ROS2 Python 패키지를 빌드하면 perception_node 실행 스크립트가 다음과 같이 생성됨.

```
#!/usr/bin/python3
```

## 해결

Dockerfile에서 venv 환경에 colcon을 추가 설치하도록 수정.

```
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel \
      ↓
    (추가)
    colcon-core \
    colcon-common-extensions
```

```
ENV PATH="/opt/venv/bin:$PATH"
```

설정을 통해 /opt/venv/bin/colcon이 우선 사용되도록 구성.

---

# 25. teleop 토픽 발행

## 파일

`scripts/twist_mux_node.py`

## 문제

teleop로 속도 명령을 내려도 로봇이 바로 정지함

## 원인

```py
if (now - self.last_teleop_time <= self.cmd_timeout):
```

teleop는 키보드 입력을 할 때만 발행하기에 추가 명령이 없으면 토픽 발행이 없어 timeout이 됨

## 해결

```py
# 신규 변수
self.teleop_received = False

# 조건 변경
if self.teleop_received:
```

---

# 26. MuJoCo Viewer GLFW X11 초기화 오류

## 파일

`mujoco_ros2_bridge.py`

## 에러코드

Docker에서 robot 컨테이너 실행 시 MuJoCo Viewer가 정상적으로 실행되지 않고 다음 오류 발생.

```
python3: /builds/florianrhiem/pyGLFW/glfw-3.4/src/x11_init.c:1099: _glfwGrabErrorHandlerX11: Assertion `_glfw.x11.errorHandler == NULL' failed.
```

Docker 컨테이너 내부에서 GLFW와 MuJoCo Viewer를 단독으로 실행하면 정상적으로 동작함.

```
python3 -c "import glfw;print(glfw.init())"
python3 -c "import mujoco;print(mujoco.__version__)"
python3 -c "import mujoco;print('mujoco OK')"
python3 -c "import mujoco.viewer;print('viewer OK')"
python3 -c "import mujoco;import mujoco.viewer;print('both OK')"
python3 -c "import mujoco;m=mujoco.MjModel.from_xml_path('/workspace/worlds/patrol_20x20_factory.xml');d=mujoco.MjData(m);import mujoco.viewer;v=mujoco.viewer.launch_passive(m,d);v.sync();import time;time.sleep(30)"
```

따라서 Docker의 DISPLAY 설정이나 X11 서버 자체의 문제는 아님.

## 원인

mujoco_ros2_bridge.py에서 MuJoCo Viewer와 카메라용 Offscreen Renderer를 동시에 사용하고 있었음.

기존 실행 순서는 다음과 같음.

```
camera_render_worker 시작
        ↓
mujoco.Renderer() 생성
        ↓
MuJoCo Viewer 생성
        ↓
mujoco.viewer.launch_passive()
        ↓
GLFW / X11 초기화
```

camera_render_worker()는 별도 스레드에서 mujoco.Renderer()를 생성하고 RGB/Depth 이미지를 렌더링함.

`renderer=mujoco.Renderer(...)`

동시에 메인 스레드에서는 MuJoCo Viewer를 생성함.

`with mujoco.viewer.launch_passive(model,data) as viewer:`

하나의 Python 프로세스 내부에서 Offscreen Rendering과 GLFW 기반 Interactive Viewer가 서로 다른 스레드에서 초기화되면서 GLFW/X11 초기화 순서에 따라 X11 error handler가 충돌하는 문제가 발생함.

오류의 핵심인

`_glfw.x11.errorHandler == NULL`

은 X11 연결 자체가 실패했다는 의미가 아니라, GLFW가 이미 설정된 X11 error handler와 충돌하는 상황에서 assertion이 발생한 것으로 판단됨.

실제로 컨테이너 내부의 환경은 정상적으로 구성되어 있었음.

```yml
DISPLAY=host.docker.internal:0.0
glfw.init() → 1
MuJoCo Viewer 단독 실행 → 정상
```

## 해결

MuJoCo Viewer를 먼저 초기화한 후 카메라용 Offscreen Renderer thread를 시작하도록 초기화 순서를 변경.

기존:

```py
render_thread=threading.Thread(target=camera_render_worker,args=(node,is_running_flag),daemon=True)
render_thread.start()

with mujoco.viewer.launch_passive(model,data) as viewer:
```

수정:

```py
with mujoco.viewer.launch_passive(model,data) as viewer:
    ...

    render_thread=threading.Thread(target=camera_render_worker,args=(node,is_running_flag),daemon=True)
    render_thread.start()
```

즉 초기화 순서를 다음과 같이 변경함.

```
MuJoCo Viewer
    ↓
GLFW / X11 초기화
    ↓
Camera Offscreen Renderer
    ↓
RGB / Depth Rendering
```

이를 통해 MuJoCo Viewer의 GLFW/X11 초기화를 먼저 완료한 뒤 별도의 Camera Renderer를 실행하도록 변경하여 GLFW X11 assertion 오류를 해결함.

---

# 27. TTC 로봇 속도 처리 방식 수정

## 파일

`ttc_node.py`

## 문제

기존 TTC 계산에서는 `/odom`에서 가져온 로봇 속도를 `robot_vel`로 사용하고 있었음.

```py
robot_vel = (self.robot_vx, self.robot_vy)
```

로봇 속도는 `/odom`의 `twist.linear`에서 가져오고 있었음.

```py
def _odom_callback(self, msg: Odometry) -> None:
    self.robot_vx = msg.twist.twist.linear.x
    self.robot_vy = msg.twist.twist.linear.y
```

이 상태에서 지게차 및 사람의 EKF 속도와 로봇의 속도를 그대로 `closing_speed` 계산에 사용하면서 상대 속도가 잘못 계산될 수 있는 문제가 발생함.

현재 `base_link` 좌표계에서 로봇 자신은 항상 원점 `(0, 0)`에 고정되어 있음.

또한 `fusion_node_3d.py`의 EKF로 추정된 객체의 `velocity`는 로봇 기준의 상대 속도로 사용되고 있음.

따라서 로봇과 객체의 TTC를 계산할 때 `/odom`에서 가져온 로봇 속도를 별도로 적용하면 상대 속도가 중복으로 계산될 수 있음.

## 원인

기존 코드는 로봇과 객체의 상대 속도를 계산할 때 다음과 같이 처리함.

```py
robot_pos = (0.0, 0.0)
robot_vel = (self.robot_vx, self.robot_vy)
```

이후 `_calculate_ttc()`에서 다음과 같이 상대 속도를 계산함.

```py
rel_vx = vel_b[0] - vel_a[0]
rel_vy = vel_b[1] - vel_a[1]
```

따라서 로봇-사람 및 로봇-지게차 TTC 계산에서는 객체의 EKF 속도에서 `/odom`의 로봇 속도를 다시 빼게 됨.

하지만 `base_link` 좌표계에서는 로봇 자체가 항상 `(0, 0)`에 위치하고 있으며, EKF에서 추정된 객체의 속도는 이미 로봇 기준의 상대 속도로 사용되고 있음.

따라서 `/odom`의 로봇 속도를 다시 적용하면 상대 속도를 잘못 계산할 수 있음.

## 해결

로봇과의 TTC 계산에서는 로봇을 `base_link` 좌표계의 원점에 고정된 객체로 취급하도록 변경함.

기존:
```py
robot_vel = (self.robot_vx, self.robot_vy)
```

수정:
```py
robot_vel = (0.0, 0.0)
```

즉 로봇과 객체 사이의 TTC 계산에서는 로봇의 `/odom` 속도를 별도로 사용하지 않고, EKF에서 추정된 객체의 상대 속도를 그대로 사용하도록 변경함.

```py
robot_pos = (0.0, 0.0)
robot_vel = (0.0, 0.0)
```

이 경우 상대 속도 계산은 다음과 같이 동작함.

```py
rel_vx = vel_b[0] - 0.0
rel_vy = vel_b[1] - 0.0
```

따라서 객체의 EKF 속도가 그대로 로봇에 대한 상대 속도로 사용됨.

## 좌표계 기준

현재 TTC 계산의 기준은 `base_link`임.

```
    base_link 좌표계

            +Y
             ↑
             |
             |
             ●────────→ +X
           Robot
          (0, 0)
```

로봇은 `base_link` 좌표계에서 항상 원점 `(0, 0)`에 위치함.

```
로봇 위치 = (0, 0)
로봇 속도 = (0, 0)

사람/지게차 위치 = EKF 추정 위치
사람/지게차 속도 = EKF 추정 상대 속도
```

따라서 로봇과 객체의 TTC 계산은 다음과 같이 동작함.

```
    로봇
    (0, 0)
      │
      │ 상대 위치
      ↓
    객체

    로봇 속도 = 0
    객체 속도 = EKF 상대 속도
            ↓
    closing_speed 계산
            ↓
    TTC 계산
```

## 결과

로봇이 Teleop 또는 Nav2로 주행 중이더라도 `base_link` 기준에서는 로봇이 원점 `(0, 0)`에 고정되어 있음.

따라서 로봇-사람 및 로봇-지게차 TTC 계산에서는 로봇의 `/odom` 속도를 별도로 적용하지 않고 객체의 EKF 상대 속도를 기준으로 접근 속도를 계산하도록 수정함.

이를 통해 로봇 속도와 객체 속도를 중복으로 적용하여 `closing_speed`가 잘못 계산되는 문제를 제거함.