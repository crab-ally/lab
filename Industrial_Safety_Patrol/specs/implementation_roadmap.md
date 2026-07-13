# AI 위험예측 안전 순찰 로봇 구현 난이도 순서

---

# Level 0. MuJoCo 기본 시뮬레이션 구축

## 1. MuJoCo 산업 현장 환경 구성 ⭐☆☆☆☆

난이도: ★☆☆☆☆

구현 내용:

- 공장 바닥, 벽, 장애물 배치
- 작업자 더미 모델 추가
- 화재 영역 추가
- 지게차 모델 추가
- 마찰력, 질량, 조명 설정

필요 기술:

- MJCF XML
- MuJoCo geom/body/joint
- 물리 파라미터 설정  

결과:

    산업 현장 시뮬레이션 환경 완성

---

## 2. 로봇 기본 이동 제어 ⭐☆☆☆☆

난이도: ★☆☆☆☆

구현 내용:

- TurtleBot3 모델 추가
- Differential Drive 제어
- Wheel actuator 연결
- Python 제어 스크립트 작성

예:

    data.ctrl[0] = left_velocity
    data.ctrl[1] = right_velocity

결과:

    MuJoCo 환경에서 로봇 이동 가능

---

# Level 1. 센서 구현

## 3. LiDAR 센서 추가 ⭐⭐☆☆☆

난이도: ★★☆☆☆
 
구현 내용:

- MuJoCo Rangefinder 추가
- LaserScan 데이터 생성
- 장애물 거리 측정

데이터 흐름:

    MuJoCo LiDAR
            ↓
    Python Bridge
            ↓
    ROS2 /scan

활용:

- 장애물 감지
- SLAM 입력 데이터

---

## 4. RGB / Depth Camera 추가 ⭐⭐☆☆☆

난이도: ★★☆☆☆

구현 내용:

- RGB Camera 생성
- Depth Camera 생성
- 이미지 Publish

ROS2 Topic:

    /camera/image_raw
    /camera/depth/image

---

# Level 2. ROS2 연동

## 5. MuJoCo ↔ ROS2 Bridge ⭐⭐⭐☆☆

난이도: ★★★☆☆

가장 중요한 기반 단계

구조:

    ROS2

    /cmd_vel
          ↓

       MuJoCo

          ↓

    /odom
    /scan
    /camera/image_raw


필요 기술:

- rclpy
- sensor_msgs
- geometry_msgs
- nav_msgs

---

## 6. Teleoperation 구현 ⭐⭐⭐☆☆

난이도: ★★★☆☆

구조:

    Keyboard
        ↓
    /cmd_vel
        ↓
    Robot Control
        ↓
    MuJoCo Robot

완성 목표:

    사용자가 로봇을 직접 조종 가능

---

# Level 3. 자율주행

## 7. SLAM 지도 생성 ⭐⭐⭐⭐☆

난이도: ★★★★☆

필요 기술:

- SLAM Toolbox
- LiDAR
- TF
- Occupancy Grid Map

구조:

    LiDAR
       ↓
    SLAM Toolbox
       ↓
    Map 생성

결과:

    factory_map.yaml
    factory_map.pgm

---

## 8. Nav2 기반 자율주행 ⭐⭐⭐⭐☆

난이도: ★★★★☆

구성:

    Goal 입력
        ↓
    Planner
        ↓
    Controller
        ↓
    /cmd_vel
        ↓
    Robot 이동

필요:

- AMCL
- Costmap
- Behavior Tree

---

# Level 4. AI 위험 감지

## 9. OpenCV 기반 화재 감지 ⭐⭐⭐☆☆

난이도: ★★★☆☆

구현:

    Camera Image
          ↓
    HSV 변환
          ↓
    불꽃 색상 검출
          ↓
    Fire Detection

장점:

- 데이터셋 필요 없음
- 빠른 구현 가능

---

## 10. YOLO 기반 PPE 탐지 ⭐⭐⭐⭐☆

난이도: ★★★★☆

구성:

    Camera
       ↓
    YOLO Detection
       ↓
    Person
    Helmet
    Vest
       ↓
    Safety 판단

필요:

- PPE Dataset
- YOLO Training
- Detection Logic

---

## 11. Pose 기반 쓰러짐 감지 ⭐⭐⭐⭐☆

난이도: ★★★★☆

구성:

    Camera
       ↓
    Pose Estimation
       ↓
    Skeleton 추출
       ↓
    Body Angle 계산
       ↓
    Fall Detection

필요:

- MediaPipe Pose
- YOLO Pose
- Temporal Filtering

---

# Level 5. 고급 인지 시스템

## 12. DeepSORT 객체 추적 ⭐⭐⭐⭐⭐

난이도: ★★★★★

목적:

- 작업자 ID 유지
- 지게차 이동 추적

구성:

    YOLO Detection
            +
    Appearance Feature
            +
    Kalman Filter

            ↓

    Object Tracking


결과:

    Worker ID 1
    Worker ID 2
    Forklift ID 3

---

## 13. Camera + LiDAR Sensor Fusion ⭐⭐⭐⭐⭐

난이도: ★★★★★

구성:

    Camera
       +
    LiDAR
       +
    Odometry

          ↓

    3D Object Position


필요:

- TF
- Calibration
- Coordinate Transform

---

## 14. TTC 충돌 위험 예측 ⭐⭐⭐⭐⭐

난이도: ★★★★★

공식:

    TTC = Distance / Relative Velocity


위험 판단:

    TTC > 3초       안전

    1.5~3초         주의

    < 1.5초         위험

---

# Level 6. 시스템 통합

## 15. Safety Node 구현 ⭐⭐⭐⭐☆

난이도: ★★★★☆

구조:

    AI Detection
          ↓
    Safety Node
          ↓
    Emergency Stop


기능:

- 긴급 정지
- 경고음 출력
- LED 경고

---

## 16. Event Logger 구축 ⭐⭐⭐☆☆

난이도: ★★★☆☆

구조:

    ROS Event
        ↓
    Logger Node
        ↓
    SQLite/PostgreSQL


저장 데이터:

    Timestamp
    Robot Position
    Event Type
    Image Path

---

## 17. Monitoring Dashboard ⭐⭐⭐⭐☆

난이도: ★★★★☆

구성:

    ROS2
      ↓
    WebSocket
      ↓
    React/Grafana
      ↓
    Dashboard


표시:

- 로봇 위치
- Camera View
- 위험 이벤트
- 사고 Heatmap

---

# Level 7. 검증 및 최적화

## 18. 자동 테스트 Harness ⭐⭐⭐⭐☆

난이도: ★★★★☆

테스트:

- 화재 발생
- 작업자 쓰러짐
- 장애물 생성


구현:

    Event 생성
        ↓
    Simulation 실행
        ↓
    Detection 확인

---

## 19. TensorRT 최적화 ⭐⭐⭐⭐⭐

난이도: ★★★★★

내용:

- YOLO ONNX 변환
- TensorRT 적용
- GPU inference
- Latency 최적화

목표:

    AI Detection < 500ms

---

# 최종 추천 구현 순서

    1. MuJoCo 공장 환경 구축
            ↓
    2. TurtleBot 이동 제어
            ↓
    3. LiDAR / Camera 추가
            ↓
    4. MuJoCo ↔ ROS2 Bridge
            ↓
    5. Teleop
            ↓
    6. SLAM
            ↓
    7. Nav2 자율주행
            ↓
    8. OpenCV 화재 감지
            ↓
    9. YOLO PPE 탐지
            ↓
    10. Pose 기반 쓰러짐 감지
            ↓
    11. Safety Node
            ↓
    12. Data Logger
            ↓
    13. Dashboard
            ↓
    14. DeepSORT
            ↓
    15. Sensor Fusion
            ↓
    16. TTC 충돌 예측
            ↓
    17. TensorRT 최적화


# 프로젝트 완성 기준

| 목표 | 구현 범위 |
|---|---|
| 포트폴리오 MVP | 1 ~ 9 단계 |
| 연구실 데모 수준 | 1 ~ 13 단계 |
| 산업용 로봇 수준 | 전체 단계 |


# 현재 프로젝트 기준 다음 목표

현재 진행 상황:

- MuJoCo industrial_factory.xml 구축
- TurtleBot 모델 구성
- ROS2 경험 보유


다음 목표:

    MuJoCo ↔ ROS2 Bridge 구축

    연결 대상:

    /cmd_vel
    /odom
    /scan
    /camera/image_raw


해당 단계가 완료되면 이후 SLAM, Nav2, AI 인식 모듈을 순차적으로 연결할 수 있는 기반이 완성된다.