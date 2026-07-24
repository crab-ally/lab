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