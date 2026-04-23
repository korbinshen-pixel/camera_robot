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
    
    # RQT图像查看器节点
    rqt_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/camera_robot/rgbd_camera/image_raw']
    )
    
    return LaunchDescription([
        spawn_robot,
        keyboard_control,
        rqt_image_view
    ])