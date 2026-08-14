import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'perception_safety_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='ROS 2 Perception and Safety Pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_safety_pkg.perception_node:main',
            'fusion_node_3d = perception_safety_pkg.fusion_node_3d:main',
            'ttc_node = perception_safety_pkg.ttc_node:main',
        ],
    },
)