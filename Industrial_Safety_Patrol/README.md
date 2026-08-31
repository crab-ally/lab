# 🤖 AI 위험예측 안전 순찰 로봇 시뮬레이션

> **ROS2 Humble + MuJoCo 3.6.0 기반 산업 현장 AI 안전 순찰 시스템**

산업 현장의 안전 사각지대를 최소화하고 중대재해를 예방하기 위해, 자율주행 순찰 로봇이 디지털 트윈 환경에서 위험 상황을 실시간으로 감지하고 대응하는 AI 기반 시뮬레이션 프로젝트입니다.

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
- [실행 방법](#-실행-방법)
- [모니터링 대시보드](#-모니터링-대시보드)
- [KPI](#-kpi)
- [주요 설계 결정 사항](#-주요-설계-결정-사항)
- [향후 계획](#-향후-계획)

---

## 🎯 프로젝트 개요

본 프로젝트는 **MuJoCo 3.6.0** 기반 디지털 트윈 환경에서 자율주행 순찰 로봇을 구현하고, 다양한 AI 기술을 활용하여 산업 현장의 위험 요소를 탐지 및 대응하는 것을 목표로 합니다.

**핵심 목표:**
- 산업 현장 안전 사각지대 최소화
- 작업자 안전사고 예방
- AI 기반 위험 상황 조기 감지 (≥95% 정확도)
- 데이터 기반 순찰 경로 최적화

---

## ✨ 주요 기능

### 🚗 자율 순찰 (Autonomous Patrol)
- **Waypoint 기반 자율주행** — 지정된 경로를 따라 공장 내부 자동 순찰
- **SLAM Mapping** — `slam_toolbox` (Karto 기반 수정)를 활용한 실시간 지도 생성
- **Nav2 자율 탐색** — 저장된 지도 기반 경로 계획 및 장애물 회피
- **자동 탐색 주행** — `auto_explore_for_mapping.py`로 미지 영역 자동 탐색 후 지도 저장
- **수동 조종 (Teleop)** — 키보드 원격 조종 지원 (`i`전진, `j`좌회전, `l`우회전, `k`정지)
- **Twist Mux** — 자율주행/원격 조종 명령 우선순위 중재

### 🔍 충돌 위험 감지 (Collision Risk Detection)
- **Multi Object Tracking (MOT)** — DeepSORT 기반 다중 객체 추적
- **TTC (Time To Collision) 계산** — Kalman Filter 기반 충돌 예상 시간 실시간 산출
- **Emergency Stop (E-Stop)** — 위험 감지 시 즉시 정지 명령 전달
- **포크리프트 컨트롤러** — 산업 차량과의 충돌 시나리오 시뮬레이션

### 🦺 비전 AI 안전 점검 (Vision AI Safety Inspection)
- **PPE 착용 여부 확인** — YOLOv8n 커스텀 모델(`ppe_forklift_yolov8n`)로 안전모·안전조끼 검출
- **Sensor Fusion** — `fusion_node_3d.py`를 통한 RGB + Depth 데이터 3D 융합
- **경고 시스템** — PPE 미착용 작업자 감지 시 경고

### 🔥 화재 감지 (Fire Detection)
- **RGB-Depth 카메라 융합** — 색상만으론 오탐 가능성이 있어 Depth 거리 분산값으로 구조물 구분
- **`fire_detection_node`** — RGB 이미지 기반 불꽃 색상 탐지
- **`fire_fusion_node`** — RGB + Depth 데이터 융합 및 최종 판정
- **비상 사이렌 알림** — 관제 시스템 자동 경보

### 🧍 쓰러짐 감지 (Fall Detection)
- **YOLOv8n-Pose 모델** — 인체 키포인트 기반 자세 추정
- **실시간 낙상 감지** — `fall_detection_node.py`로 작업자 쓰러짐 즉시 감지

### 📊 데이터 분석 (Safety Analytics)
- **이벤트 로깅** — `event_logger_pkg`를 통해 모든 안전 이벤트를 PostgreSQL(TimescaleDB)에 저장
- **CSV / JSON 로그** — 사고 발생 위치 및 상세 정보 기록
- **Heatmap 생성** — Canvas 기반 위험 구역 시각화
- **웹 대시보드** — 실시간 원격 관제 스트리밍

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              Remote Dashboard (localhost:8080)        │
│          React 단일 파일 웹앱 (nginx)                 │
└──────────────────────┬──────────────────────────────┘
                       │ WebSocket (port 8765)
┌──────────────────────▼──────────────────────────────┐
│                  ws_bridge.py                        │
│         ROS2 Topic → WebSocket 브리지               │
└──────────────────────┬──────────────────────────────┘
                       │ ROS2 Topics (DDS, ROS_DOMAIN_ID=50)
┌──────────────────────▼──────────────────────────────┐
│              ROS2 Humble (Docker Network)            │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ perception_  │  │ fire_        │  │ fall_     │  │
│  │ safety_pkg   │  │ detection_   │  │ detection │  │
│  │              │  │ pkg          │  │ _node.py  │  │
│  │ - perception │  │ - fire_      │  │           │  │
│  │ - fusion_3d  │  │   detection  │  │ YOLOv8n   │  │
│  │ - ttc_node   │  │ - fire_fusion│  │ Pose      │  │
│  │ - fl_ctrl    │  └──────────────┘  └───────────┘  │
│  └──────────────┘                                    │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ event_logger │  │ Nav2 / SLAM  │                  │
│  │ _pkg         │  │ slam_toolbox │                  │
│  │              │  │ (Karto 수정) │                  │
│  │ PostgreSQL   │  └──────────────┘                  │
│  │ TimescaleDB  │                                    │
│  └──────────────┘                                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          MuJoCo 3.6.0 Simulation Bridge              │
│         mujoco_ros2_bridge.py                        │
│                                                      │
│  Digital Twin: 공장 환경 (10x10 / 20x20)             │
│  Robot: TurtleBot3 (LiDAR 360°, RGB-D Camera)        │
└─────────────────────────────────────────────────────┘
```

### ROS2 주요 토픽

| 토픽 | 설명 |
|------|------|
| `/ttc_alerts` | TTC 충돌 위험 경보 |
| `/detections_2d` | YOLO 탐지 결과 |
| `/tracks_3d` | 3D 추적 정보 |
| `/odom` | 로봇 위치/속도 |
| `/cmd_vel_nav` | Nav2 이동 명령 |
| `/cmd_vel_teleop` | 원격 조종 명령 |

---

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| **시뮬레이션** | MuJoCo 3.6.0 |
| **ROS2** | Humble Hawksbill |
| **언어** | Python 3.10+ |
| **AI / Vision** | YOLOv8n (Ultralytics), OpenCV 4.10 |
| **추적** | DeepSORT (deep-sort-realtime), Kalman Filter (filterpy) |
| **SLAM / Navigation** | slam_toolbox (Karto 수정), Nav2 |
| **데이터베이스** | PostgreSQL 15 + TimescaleDB |
| **수치 연산** | NumPy 1.26.4, SciPy 1.13.1 |
| **대시보드** | React (CDN), WebSocket, nginx |
| **컨테이너** | Docker, Docker Compose |

---

## 📁 프로젝트 구조

```
Industrial_Safety_Patrol/
├── Dockerfile                    # 멀티 스테이지 빌드 (robot / db / dashboard)
├── docker-compose.yml            # 전체 서비스 오케스트레이션
├── HowToUse.txt                  # 빠른 실행 명령 참조 가이드
│
├── ros2_ws/                      # ROS2 워크스페이스
│   └── src/
│       ├── perception_safety_pkg/    # 비전 AI 안전 점검
│       │   └── perception_safety_pkg/
│       │       ├── perception_node.py    # YOLOv8 PPE 탐지
│       │       ├── fusion_node_3d.py     # RGB-D 3D 센서 융합
│       │       ├── ttc_node.py           # TTC 충돌 예측
│       │       └── forklift_controller.py # 포크리프트 제어
│       ├── fire_detection_pkg/       # 화재 감지
│       │   └── fire_detection_pkg/
│       │       ├── fire_detection_node.py # RGB 불꽃 탐지
│       │       └── fire_fusion_node.py    # RGB-D 융합 최종 판정
│       ├── event_logger_pkg/         # 이벤트 로깅 (PostgreSQL)
│       │   └── event_logger_pkg/
│       │       └── event_logger_node.py
│       └── slam_toolbox/             # Karto 기반 SLAM (Inf ray 처리 수정)
│
├── scripts/                      # 핵심 실행 스크립트
│   ├── mujoco_ros2_bridge.py         # MuJoCo ↔ ROS2 브리지 (headless)
│   ├── mujoco_ros2_bridge_viewer.py  # MuJoCo ↔ ROS2 브리지 (viewer)
│   ├── twist_mux_node.py             # 이동 명령 우선순위 중재
│   ├── fall_detection_node.py        # YOLOv8-Pose 낙상 감지
│   ├── auto_explore_for_mapping.py   # 자동 탐색 주행 + 지도 생성
│   ├── save_factory_map.py           # 지도 저장
│   └── dataset_generator/            # 학습 데이터셋 생성 도구
│
├── dashboard/                    # 웹 모니터링 대시보드
│   ├── index.html                    # React CDN 단일 파일 웹앱
│   ├── ws_bridge.py                  # ROS2 → WebSocket 브리지 (port 8765)
│   └── README.md
│
├── worlds/                       # MuJoCo 공장 환경
│   ├── 10x10_industrial_factory.xml
│   └── 20x20_industrial_factory.xml
│
├── models/                       # AI 모델 및 로봇 모델
│   ├── turtlebot_patrol.xml          # TurtleBot3 MuJoCo 모델
│   ├── ppe_forklift_yolov8n/         # PPE + 포크리프트 탐지 모델
│   └── fall_yolov8n_pose/            # 낙상 감지 Pose 모델
│
├── config/                       # ROS2 설정 파일
│   ├── nav2_params.yaml              # Nav2 파라미터
│   ├── slam_toolbox_factory.yaml     # SLAM Toolbox 설정
│   └── nav2_default_view.rviz        # RViz 기본 뷰 설정
│
├── launch/                       # ROS2 런치 파일
│   └── robot_state_publisher.launch.py
│
├── urdf/                         # 로봇 URDF 모델
├── maps/                         # 저장된 공장 지도
├── data/                         # 로그 / 처리 데이터
├── datasets/                     # 학습 데이터셋
├── results/                      # 분석 결과
├── docs/                         # 문서
│   ├── DECISIONS.md                  # 주요 설계 결정 기록
│   ├── TROUBLESHOOTING.md            # 트러블슈팅 가이드
│   └── Future_Improvements.md
└── specs/                        # 기획 문서
    ├── PRD.md
    ├── PseudoCode.md
    └── implementation_roadmap.md
```

---

## 🚀 시작하기

### 사전 요구사항

- [Docker](https://www.docker.com/) & Docker Compose
- X11 서버 (MuJoCo viewer 사용 시) — Windows: [VcXsrv](https://sourceforge.net/projects/vcxsrv/) 또는 X410

### 설치

```bash
git clone https://github.com/your-id/Industrial_Safety_Patrol.git
cd Industrial_Safety_Patrol

# Docker 이미지 빌드
docker compose build
```

---

## ▶ 실행 방법

> 전체 실행 명령은 [`HowToUse.txt`](./HowToUse.txt)에서 빠르게 확인할 수 있습니다.

### 1단계 — MuJoCo 시뮬레이터 + ROS2 브리지

```bash
# Headless 모드 (viewer 없음)
docker compose up robot
docker compose exec robot bash

# Viewer 모드 (MuJoCo 창 표시, X11 필요)
docker compose up robot_viewer
docker compose exec robot_viewer bash
```

### 2단계 — SLAM / 자율 탐색

```bash
# RViz + SLAM (실시간 지도 생성)
docker compose up rviz_slam

# 자동 탐색 주행 + 지도 저장
docker compose run --rm auto_explore_map
```

### 3단계 — Nav2 자율 순찰

```bash
# RViz + Nav2 (저장된 지도 기반 자율 순찰)
docker compose up rviz_nav2
```

### 4단계 — AI 안전 감지 모듈

```bash
# 화재 감지 (fire_detection_node + fire_fusion_node)
cd ros2_ws && colcon build --packages-select fire_detection_pkg
docker compose up fire_detection

# PPE + 충돌 예측 + Sensor Fusion + TTC
cd ros2_ws && colcon build --packages-select perception_safety_pkg
docker compose up perception_safety

# 낙상 감지 (YOLOv8-Pose)
docker compose up fall_detection
```

### 5단계 — 데이터 로깅

```bash
# PostgreSQL 이벤트 로거
cd ros2_ws && colcon build --packages-select event_logger_pkg
docker compose up event_logger

# DB 접속 및 이벤트 조회
docker compose exec db psql -U admin -d patrol_db
SELECT id, timestamp, event_type, severity, track_id, metadata
FROM safety_events ORDER BY id DESC;
\q
```

### 6단계 — 모니터링 대시보드

```bash
# WebSocket 브리지 + 대시보드 동시 실행
docker compose --profile dashboard up

# 브라우저 접속
# http://localhost:8080
```

### 수동 조종 (Teleop)

```bash
docker compose run --rm teleop
# i: 전진  |  k: 정지  |  j: 좌회전  |  l: 우회전
```

---

## 📡 모니터링 대시보드

`http://localhost:8080` 접속 시 다음 정보를 실시간으로 확인할 수 있습니다:

| 패널 | 내용 |
|------|------|
| **Factory Map** | 공장 SVG 맵 + 로봇 실시간 위치 |
| **Event Feed** | TTC 경보, PPE 위반 이벤트 스트림 |
| **Statistics** | 총 이벤트, 위험 수준별 분포 |
| **TTC Alerts** | 충돌 예상 시간 실시간 표시 |
| **Robot Odometry** | 로봇 좌표 및 속도 |
| **Heatmap** | Canvas 기반 사고 발생 위험 히트맵 |

**WebSocket 브리지** (`ws_bridge.py`)가 ROS2 토픽을 포트 `8765`를 통해 브라우저로 전달합니다.

---

## 📊 KPI

| 지표 | 목표 |
|------|------|
| 위험 감지 정확도 | ≥ 95% |
| 충돌 사고 건수 | 0 건 |
| 순찰 가용성 | 24시간 |
| 엣지 응답 시간 | ≤ 500ms |
| 항법 정확도 | ±10mm |

---

## 🔧 주요 설계 결정 사항

자세한 내용은 [`docs/DECISIONS.md`](./docs/DECISIONS.md) 참고

| 항목 | 결정 내용 |
|------|----------|
| **LiDAR Ray 수** | 초기 36개(10° 해상도) → SLAM 정밀도 문제로 360개(1° 해상도)로 변경 |
| **SLAM** | Karto 기반 SLAM Toolbox의 `AddScan()` 수정 — `Inf` 측정값을 최대거리 Free 공간으로 처리 |
| **화재 감지** | RGB 단독 → RGB-Depth 융합 (소화전 등 오탐 방지, Depth 분산값으로 구조물 구분) |
| **Robot State Publisher** | Docker CMD 직접 전달 → Launch 파일 기반으로 변경 (XML 파싱 오류 방지) |

---

## 🔭 향후 계획

- [ ] LLM 기반 위험 상황 자연어 분석 및 보고서 자동 생성
- [ ] 다중 로봇 협업 순찰 (Multi-Agent)
- [ ] 클라우드 기반 관제 시스템 연동
- [ ] 실제 자율주행 로봇 하드웨어 적용
- [ ] YOLO 기반 객체 탐지 고도화 (YOLOv10+)
- [ ] 5G 실시간 스트리밍 시뮬레이션

---

## 📄 라이선스

본 프로젝트는 학술 및 연구 목적으로 제작되었습니다.

This project is intended for academic and research purposes.

