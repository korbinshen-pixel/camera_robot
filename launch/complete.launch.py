# 创建完整的启动文件
# 创建 src/camera_robot/launch/complete.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory('camera_robot')
    
    # 包含spawn_robot.launch.py
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'spawn_robot.launch.py')
        ])
    )
    
    # 键盘控制节点
    keyboard_control = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        prefix='xterm -e',
        output='screen',
        remappings=[
            ('/cmd_vel', '/camera_robot/cmd_vel')
        ]
    )
    
    # 托盘检测节点
    pallet_detector = Node(
        package='camera_robot',
        executable='pallet_detector',
        name='pallet_detector',
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
        spawn_robot,
        keyboard_control,
        pallet_detector,
        rqt_image_view
    ])