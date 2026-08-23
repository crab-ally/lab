import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'event_logger_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='ROS2 Safety Event Logger — SQLite/PostgreSQL 이벤트 저장 패키지',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'event_logger_node = event_logger_pkg.event_logger_node:main',
        ],
    },
)
