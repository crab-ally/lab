import sys
if sys.prefix == '/opt/venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspace/ros2_ws/install/perception_safety_pkg'
