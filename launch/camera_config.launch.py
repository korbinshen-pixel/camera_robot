import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 托盘检测节点
    pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector',
        name='pallet_detector',
        output='screen',
        parameters=[
            {'debug_mode': False}
        ]
    )
    
    # 简化的托盘检测节点（备选）
    simple_pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector_simple',
        name='simple_pallet_detector',
        output='screen'
    )
    
    # RQT图像查看器（查看检测结果）
    rqt_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/camera_robot/annotated_image']
    )
    
    return LaunchDescription([
        # pallet_detector,
        # # 可以选择运行其中一个检测器
        simple_pallet_detector,
        # rqt_image_view
    ])