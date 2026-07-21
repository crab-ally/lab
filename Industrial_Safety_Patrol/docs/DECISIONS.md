# 1. Lidar 개수

**결정:** 실제 LDS-01 LiDAR(360°, 1° 해상도)를 단순화하여 MuJoCo에서는 36개 Ray(10° 해상도)로 구현

**이유**
- 모든 방향의 Ray를 구현할 경우 센서 계산량과 시뮬레이션 부하가 증가
- 장애물 감지, SLAM 입력, Navigation 기능 검증에 필요한 공간 정보는 유지하면서 계산 효율을 높이기 위해 단순화하여 구현

---

# 2. Robot State Publisher 실행 방식

**결정:** robot_state_publisher를 직접 실행하는 방식 대신 별도의 ROS2 Launch 파일을 생성하여 URDF를 전달하는 구조로 변경

**이유**
- robot_state_publisher는 실행 시 robot_description parameter로 URDF 정보가 반드시 필요함
- Docker Compose command에서 -p robot_description:=URDF 내용 형태로 직접 전달할 경우 XML 내부의 <, >, 개행 문자 때문에 ROS2 parameter parser 오류 발생
- Launch 파일에서 URDF를 읽어 robot_description parameter로 전달하면 안정적으로 TF 정보를 생성 가능
ROS2 표준 방식에 맞춰 유지보수성과 확장성을 높이기 위해 Launch 기반 구조로 변경