# 🤖 AI Safety Patrol Robot Simulation

> **MuJoCo 3.6.0 기반 통합 AI 안전 순찰 로봇 시뮬레이션 시스템**

산업 현장의 안전 사각지대를 최소화하고 중대재해를 예방하기 위해, 자율주행 순찰 로봇이 디지털 트윈 환경에서 위험 상황을 실시간으로 감지하고 대응하는 AI 기반 시뮬레이션 프로젝트입니다.

---

## Overview

본 프로젝트는 MuJoCo 기반의 디지털 트윈 환경에서 자율주행 순찰 로봇을 구현하고, 다양한 AI 기술을 활용하여 산업 현장의 위험 요소를 탐지 및 대응하는 것을 목표로 합니다.

주요 기능은 다음과 같습니다.

- 자율 순찰 및 경로 재탐색
- 작업자 넘어짐 감지
- 충돌 위험 예측 및 비상 정지
- PPE(안전모, 안전조끼) 착용 여부 확인
- 화재 및 고온 이상 감지
- 위험 상황 자동 로깅
- 안전 리스크 맵(Heatmap) 생성
- 원격 관제 스트리밍

---

# Project Goals

- 산업 현장의 안전 사각지대 최소화
- 작업자 안전사고 예방
- AI 기반 위험 상황 조기 감지
- 디지털 트윈 기반 안전 관리 시스템 구축
- 데이터 기반 순찰 경로 최적화

---

# Key Features

## Autonomous Patrol

- 자가 진단(Self Diagnosis)
- Waypoint 기반 자율주행
- SLAM Map 기반 순찰
- 장애물 회피
- 실시간 경로 재탐색

---

## Collision Risk Detection

- Multi Object Tracking(MOT)
- Kalman Filter 기반 객체 추적
- TTC(Time To Collision) 계산
- 위험 발생 시 Emergency Stop(E-Stop)
- 작업자 넘어짐 감지

---

## Vision AI Safety Inspection

- PPE 착용 여부 확인
- 안전모 검출
- 안전조끼 검출
- 음성 경고 시스템

---

## Fire Detection

- 열화상 센서 기반 이상 온도 탐지
- 불꽃 감지
- 비상 사이렌 동작
- 관제 시스템 알림

---

## Safety Analytics

- 위험 이벤트 자동 저장
- CSV / JSON 로그 생성
- 사고 발생 위치 기록
- Heatmap 생성
- 위험 구역 분석
- 순찰 경로 최적화

---

# System Architecture

```
                   +-----------------------+
                   |  Remote Dashboard     |
                   |  Monitoring / Control |
                   +-----------+-----------+
                               ^
                               |
                         Video / Telemetry
                               |
+------------------------------------------------------------+
|                 AI Patrol Robot (Edge)                     |
|------------------------------------------------------------|
| Autonomous Navigation                                      |
| Collision Prediction                                       |
| Vision AI                                                  |
| Fire Detection                                             |
| Risk Logging                                               |
+----------------------------+-------------------------------+
                             |
                             |
                    MuJoCo 3.6.0 Simulation
                             |
                    Digital Twin Environment
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Simulation | MuJoCo 3.6.0 |
| Computer Vision | OpenCV |
| Navigation | SLAM |
| Tracking | Kalman Filter |
| Sensor | LiDAR, RGB-D Camera, IMU |
| Data | CSV, JSON |
| Streaming | 5G / LTE Simulation |

---

# Project Structure

```
Industrial_Safety_Patrol
├─ data
│  ├─ logs
│  ├─ processed
│  └─ raw
├─ docs
├─ models
├─ notebooks
├─ results
├─ scripts
└─ worlds
```

---

# Functional Requirements

## 1. Autonomous Patrol

- Self Diagnosis
- Waypoint Navigation
- SLAM Mapping
- Dynamic Path Planning

## 2. Collision Prevention

- Multi Object Tracking
- Collision Prediction (TTC)
- Emergency Stop
- Fall Detection

## 3. Vision AI

- PPE Detection
- Helmet Detection
- Safety Vest Detection

## 4. Fire Detection

- Thermal Monitoring
- Flame Detection
- Emergency Alarm

## 5. Data Analytics

- Event Logging
- Snapshot Capture
- Heatmap Generation
- Risk Analysis

---

# KPI

| Metric | Target |
|---------|--------|
| Hazard Detection Accuracy | ≥95% |
| Collision Incidents | 0 |
| Patrol Availability | 24 Hours |
| Edge Response Time | ≤500ms |
| Navigation Accuracy | ±10mm |

---

# Getting Started

## Requirements

- Python 3.10+
- MuJoCo 3.6.0
- OpenCV
- NumPy

## Installation

```bash
git clone https://github.com/your-id/AI-Safety-Patrol-Robot.git

cd AI-Safety-Patrol-Robot

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

---

# Output

시뮬레이션 실행 시 다음과 같은 결과를 확인할 수 있습니다.

- 자율 순찰
- 장애물 회피
- 작업자 추적
- PPE 검출
- 화재 감지
- 이벤트 로그 저장
- 위험 Heatmap 생성
- 관제 대시보드 스트리밍

---

# Future Work

- ROS2 연동
- 실제 자율주행 로봇 적용
- YOLO 기반 객체 탐지 고도화
- LLM 기반 위험 상황 분석
- 다중 로봇 협업 순찰
- 클라우드 기반 관제 시스템

---

# License

This project is intended for academic and research purposes.