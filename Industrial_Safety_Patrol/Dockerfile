# ================================================================
# AI Safety Patrol Robot
# ROS2 Humble + MuJoCo 3.6.0 + AI Vision Environment
# ================================================================

# ================================================================
# 1. Robot Image
# ================================================================

FROM osrf/ros:humble-desktop-full AS robot

# ================================================================
# 2. System Package Installation
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
    # ------------------------------------------------------------
    # MuJoCo / OpenGL / GLFW
    # ------------------------------------------------------------
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
    # ------------------------------------------------------------
    # ROS2
    # ------------------------------------------------------------
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-teleop-twist-keyboard \
    \
    # ------------------------------------------------------------
    # Navigation / SLAM
    # ------------------------------------------------------------
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-diagnostic-updater \
    ros-humble-nav2-msgs \
    ros-humble-nav2-common \
    ros-humble-slam-toolbox \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# 3. Workspace
# ================================================================

WORKDIR /workspace

# ================================================================
# 4. Python Virtual Environment
#
# --system-site-packages
#   → ROS2 Python 패키지(rclpy, cv_bridge 등)를 사용
#
# venv site-packages
#   → Ultralytics, DeepSORT, MuJoCo 등
# ================================================================

RUN python3 -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN which python3 && \
    python3 --version && \
    which pip && \
    pip --version

# ================================================================
# 5. Python Package Manager
# ================================================================

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel \
    colcon-core \
    colcon-common-extensions

# ================================================================
# 6. AI / Simulation Python Packages
# ================================================================

RUN pip install \
    mujoco==3.6.0 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    opencv-python-headless==4.10.0.84 \
    ultralytics \
    pandas \
    matplotlib \
    psycopg2-binary \
    filterpy \
    pytest \
    deep-sort-realtime

# ================================================================
# 7. Verify Python Environment
# ================================================================

RUN python3 -c "import numpy; print('NumPy:', numpy.__version__)" && \
    python3 -c "import scipy; print('SciPy:', scipy.__version__)" && \
    python3 -c "import mujoco; print('MuJoCo:', mujoco.__version__)" && \
    python3 -c "import cv2; print('OpenCV:', cv2.__version__)" && \
    python3 -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)" && \
    python3 -c "import deep_sort_realtime; print('DeepSORT: OK')"

# ================================================================
# 8. MuJoCo Rendering
# ================================================================

ENV MUJOCO_GL=glfw
ENV PYTHONUNBUFFERED=1

# ================================================================
# 9. ROS2 Environment
# ================================================================

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo 'export PATH="/opt/venv/bin:$PATH"' >> /root/.bashrc

# ================================================================
# 10. Project
# ================================================================

COPY . /workspace

# ================================================================
# 11. Default Command
# ================================================================

CMD ["bash"]

# ================================================================
# 12. Database Image
# ================================================================

FROM timescale/timescaledb:latest-pg15 AS db

# ================================================================
# 13. Dashboard Image
# ================================================================

FROM grafana/grafana:latest AS dashboard