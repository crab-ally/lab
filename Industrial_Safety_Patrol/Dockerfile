# ================================================================
# AI Safety Patrol Robot
# ROS2 Humble + MuJoCo 3.6.0 + AI Vision Environment
# ================================================================

FROM osrf/ros:humble-desktop-full AS robot

# ================================================================
# 1. System Package Installation
# ================================================================

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    vim \
    \
    # MuJoCo Viewer (GLFW) + Headless Rendering
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    libx11-6 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    \
    # ROS2 Packages
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-teleop-twist-keyboard \
    \
    # Navigation
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-diagnostic-updater \
    ros-humble-nav2-msgs \
    ros-humble-nav2-common \
    ros-humble-slam-toolbox \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    && apt-get install --only-upgrade -y \
      ros-humble-navigation2 \
      ros-humble-nav2-bringup \
      ros-humble-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# 2. Workspace
# ================================================================

WORKDIR /workspace

# ================================================================
# 3. Python Virtual Environment
# ================================================================

RUN python3 -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# ================================================================
# 4. Python Packages
# ================================================================

RUN pip install \
    mujoco==3.6.0 \
    numpy==1.26.4 \
    scipy \
    opencv-python==4.10.0.84 \
    ultralytics \
    pandas \
    matplotlib \
    psycopg2-binary \
    filterpy \
    pytest

# ================================================================
# 5. MuJoCo Rendering
# ================================================================

ENV MUJOCO_GL=glfw
ENV PYTHONUNBUFFERED=1

# ================================================================
# 6. ROS2 Environment
# ================================================================

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# ================================================================
# 7. Copy Project
# ================================================================

COPY . /workspace

# ================================================================
# 8. Default Command
# ================================================================

CMD ["bash"]

# ================================================================
# 9. Database Service
# ================================================================
FROM timescale/timescaledb:latest-pg15 AS db

# ================================================================
# 10. Dashboard Service
# ================================================================
FROM grafana/grafana:latest AS dashboard