import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'fire_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Fire detection and 3D sensor fusion package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fire_detection_node = fire_detection_pkg.fire_detection_node:main',
            'fire_fusion_node = fire_detection_pkg.fire_fusion_node:main',
        ],
    },
)