# 1. 베이스 이미지 설정 (ROS2 및 Python 3.10 포함)
FROM osrf/ros:humble-desktop-full

# 2. 필수 시스템 패키지 설치 (MuJoCo 및 OpenCV 의존성)
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. Python 의존성 설치 (소스 사양 기준) [3, 4]
# mujoco: 물리 엔진, ultralytics: YOLOv8(PPE 인식), psycopg2: DB 연동
RUN pip3 install --upgrade pip
RUN pip3 install \
    mujoco==3.6.0 \
    numpy \
    opencv-python \
    ultralytics \
    pandas \
    psycopg2-binary \
    filterpy \
    pytest

# 5. 환경 변수 설정 (MuJoCo 렌더링 관련)
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

# 6. 소스 코드 복사
COPY . .

# 7. 실행 명령 (시뮬레이션 메인 루프 실행) [2]
CMD ["python3", "main_simulation.py"]