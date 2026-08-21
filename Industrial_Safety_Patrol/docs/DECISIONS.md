# Lidar 개수

**결정:** 실제 LDS-01 LiDAR(360°, 1° 해상도)를 단순화하여 MuJoCo에서는 36개 Ray(10° 해상도)로 구현

**이유**
- 모든 방향의 Ray를 구현할 경우 센서 계산량과 시뮬레이션 부하가 증가
- 장애물 감지, SLAM 입력, Navigation 기능 검증에 필요한 공간 정보는 유지하면서 계산 효율을 높이기 위해 단순화하여 구현

**이후 수정:**  SLAM이 생성한 지도가 월드의 장애물을 제대로 인식하지 못하여 360개 Ray(1° 해상도)로 변경


---

# Robot State Publisher 실행 방식

**결정:** robot_state_publisher를 직접 실행하는 방식 대신 별도의 ROS2 Launch 파일을 생성하여 URDF를 전달하는 구조로 변경

**이유**
- robot_state_publisher는 실행 시 robot_description parameter로 URDF 정보가 반드시 필요함
- Docker Compose command에서 -p robot_description:=URDF 내용 형태로 직접 전달할 경우 XML 내부의 <, >, 개행 문자 때문에 ROS2 parameter parser 오류 발생
- Launch 파일에서 URDF를 읽어 robot_description parameter로 전달하면 안정적으로 TF 정보를 생성 가능
ROS2 표준 방식에 맞춰 유지보수성과 확장성을 높이기 위해 Launch 기반 구조로 변경

---

# 화재 감지 기능

**결정:** RGB-Depth 카메라

**이유**
- RGB 카메라 만으로는 소화전 같은 색상이 일치하는 다른 물체들도 감지되는 현상 발생
- mujoco viewer 상 동적인 불꽃 표현 어려움
- Depth 카메라의 거리값의 분산치로 구조물 구분

---

# SLAM Mapping 기능 개선

**결정:** Karto 기반 SLAM Toolbox 수정

**이유**
- 기존 Karto는 Inf 라이다 측정값을 유효하지 않은 데이터로 처리하여 해당 방향을 Free 공간으로 반영하지 않음
- Inf 측정값을 LiDAR 최대 거리까지의 Free 공간으로 처리하도록 Karto의 AddScan() 로직 수정