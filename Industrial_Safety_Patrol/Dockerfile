# ================================================================
# AI Safety Patrol Robot
# ROS2 Humble + MuJoCo 3.6.0 + AI Vision Environment
# ================================================================

FROM osrf/ros:humble-desktop-full


# ================================================================
# 1. System Package Installation
# ================================================================

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    vim \
    \
    # MuJoCo Rendering Dependencies
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    \
    # ROS2 Python / Sensor Packages
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    \
    # Navigation Stack
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    \
    && rm -rf /var/lib/apt/lists/*



# ================================================================
# 2. Workspace
# ================================================================

WORKDIR /app



# ================================================================
# 3. Python Package Installation
# ================================================================

RUN pip3 install --upgrade pip


RUN pip3 install \
    mujoco==3.6.0 \
    numpy \
    scipy \
    opencv-python \
    ultralytics \
    torch \
    torchvision \
    pandas \
    matplotlib \
    psycopg2-binary \
    filterpy \
    pytest



# ================================================================
# 4. MuJoCo Rendering Environment
# ================================================================

# EGL : GPU / Headless Rendering
ENV MUJOCO_GL=egl

# Python 출력 버퍼 제거
ENV PYTHONUNBUFFERED=1



# ================================================================
# 5. ROS2 Environment
# ================================================================

# Container 실행 시 ROS2 자동 source
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc



# ================================================================
# 6. Copy Project Source
# ================================================================

COPY . /app



# ================================================================
# 7. Default Command
# ================================================================

CMD ["python3", "scripts/main_simulation.py"]