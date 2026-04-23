"""
camera_config.launch.py
单独启动传感器处理节点（不启动 Gazebo），用于调试。
  - 简化托盘检测（HSV + 深度）
  - 激光雷达监视
  - rqt 查看标注结果 / 雷达图
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # 简化托盘检测节点（订阅深度图）
    simple_pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector_simple',
        name='simple_pallet_detector',
        output='screen'
    )

    # 完整托盘检测节点（RGBD + solvePnP / 深度测距）
    pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector',
        name='pallet_detector',
        output='screen'
    )

    # 激光雷达监视节点
    lidar_viewer = Node(
        package='camera_robot',
        executable='lidar_viewer',
        name='lidar_viewer',
        output='screen'
    )

    # RQT 查看标注图像
    rqt_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/camera_robot/annotated_image']
    )

    # RQT 查看雷达俯视图
    rqt_lidar = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_lidar',
        arguments=['/camera_robot/lidar_image']
    )

    return LaunchDescription([
        # 选择其中一个检测器（取消注释以启用）：
        simple_pallet_detector,
        # pallet_detector,
        lidar_viewer,
        rqt_image_view,
        rqt_lidar,
    ])
