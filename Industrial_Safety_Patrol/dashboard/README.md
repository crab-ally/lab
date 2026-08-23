# Monitoring Dashboard

AI 위험예측 안전 순찰 로봇 실시간 모니터링 대시보드

## 구성

```
dashboard/
├── index.html     ← React CDN 단일 파일 웹 앱 (빌드 불필요)
├── ws_bridge.py   ← ROS2 → WebSocket 브리지 서버
└── README.md
```

## 실행 방법

### 1. WebSocket 브리지 서버 시작 (ROS2 환경 필요)

```bash
# ROS2 환경 소스
source /opt/ros/humble/setup.bash

# 브리지 실행
python3 /workspace/dashboard/ws_bridge.py
```

또는 docker-compose로:

```bash
docker compose up ws_bridge
```

### 2. 대시보드 웹 서버 시작

```bash
# Python 내장 서버 (로컬 테스트)
cd dashboard
python3 -m http.server 8080

# 또는 docker-compose로:
docker compose up dashboard
```

### 3. 브라우저 접속

```
http://localhost:8080
```

## 환경변수

| 변수       | 기본값    | 설명                    |
|----------|---------|-------------------------|
| WS_HOST  | 0.0.0.0 | WebSocket 서버 바인딩 주소 |
| WS_PORT  | 8765    | WebSocket 서버 포트        |

## 표시 항목

| 패널              | 내용                          |
|-----------------|-------------------------------|
| Factory Map     | 공장 SVG 맵 + 로봇 실시간 위치 |
| Event Feed      | TTC 경보, PPE 위반 이벤트 스트림 |
| Statistics      | 총 이벤트, 위험 수준별 분포     |
| TTC Alerts      | 충돌 예상 시간 실시간 표시       |
| Robot Odometry  | 로봇 좌표 및 속도               |
| Heatmap         | Canvas 기반 사고 발생 히트맵     |

## 수신 토픽

| ROS2 Topic       | 설명              |
|-----------------|-------------------|
| `/ttc_alerts`   | TTC 충돌 위험 경보  |
| `/detections_2d`| YOLO 탐지 결과     |
| `/tracks_3d`    | 3D 추적 정보       |
| `/odom`         | 로봇 위치/속도      |
