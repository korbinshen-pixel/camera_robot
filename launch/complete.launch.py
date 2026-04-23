"""
complete.launch.py
完整启动文件：
  Gazebo + 机器人 + 托盘
  → teleop_twist_keyboard（键盘控制）
  → pallet_detector（深度相机托盘检测）
  → lidar_viewer（雷达俯视图）
  → rqt_image_view（查看标注结果）
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('camera_robot')

    # 1. 启动 Gazebo + 生成机器人 + 生成托盘
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'spawn_robot.launch.py')
        ])
    )

    # 2. 键盘控制（在 xterm 窗口中运行）
    keyboard_control = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        prefix='xterm -e',
        output='screen',
        remappings=[('/cmd_vel', '/camera_robot/cmd_vel')]
    )

    # 3. 托盘检测节点（RGBD 深度相机版）
    pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector',
        name='pallet_detector',
        output='screen'
    )

    # 4. 激光雷达监视节点
    lidar_viewer = Node(
        package='camera_robot',
        executable='lidar_viewer',
        name='lidar_viewer',
        output='screen'
    )

    # 5. RQT 查看标注图像
    rqt_annotated = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_annotated',
        arguments=['/camera_robot/annotated_image']
    )

    # 6. RQT 查看雷达俯视图
    rqt_lidar = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_lidar',
        arguments=['/camera_robot/lidar_image']
    )

    return LaunchDescription([
        spawn_robot,
        keyboard_control,
        pallet_detector,
        lidar_viewer,
        rqt_annotated,
        rqt_lidar,
    ])
