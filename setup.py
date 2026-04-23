from setuptools import setup
import os
from glob import glob

package_name = 'camera_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 包含launch文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # 包含机器人模型文件
        (os.path.join('share', package_name, 'models', 'camera_robot'), 
         glob('models/camera_robot/*')),
        # 包含托盘模型文件
        (os.path.join('share', package_name, 'models', 'pallet'), 
         glob('models/pallet/*')),
        # 包含世界文件
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A robot with RGBD camera controlled by keyboard',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = camera_robot.robot_controller:main',
            'camera_viewer = camera_robot.camera_viewer:main',
            'pallet_detector = camera_robot.pallet_detector:main',
            'pallet_detector_simple = camera_robot.pallet_detector_simple:main',
        ],
    },
)