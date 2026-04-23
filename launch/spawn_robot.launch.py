import os
import random
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory('camera_robot')
    
    # Gazebo启动文件
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': os.path.join(pkg_path, 'worlds', 'empty.world'),
        }.items()
    )
    
    # 生成机器人节点
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'camera_robot',
            '-file', os.path.join(pkg_path, 'models', 'camera_robot', 'model.sdf'),
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.2'
        ],
        output='screen'
    )
    
    # 随机生成托盘位置（在小车附近1-2米范围内）
    # 生成随机的x和y坐标，避免与小车太近或太远
    pallet_x = random.uniform(-2.0, 2.0)
    pallet_y = random.uniform(-2.0, 2.0)
    
    # 确保托盘不会与小车重叠（距离至少1米）
    while abs(pallet_x) < 1.0 and abs(pallet_y) < 1.0:
        pallet_x = random.uniform(-2.0, 2.0)
        pallet_y = random.uniform(-2.0, 2.0)
    
    # 随机生成朝向角度（0-360度）
    pallet_yaw = random.uniform(0, 6.28318)  # 0到2π弧度
    
    # 生成托盘节点
    spawn_pallet = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'pallet',
            '-file', os.path.join(pkg_path, 'models', 'pallet', 'model.sdf'),
            '-x', str(pallet_x),
            '-y', str(pallet_y),
            '-z', '0.05',  # 托盘高度0.05米
            '-Y', str(pallet_yaw)
        ],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo_launch,
        spawn_robot,
        spawn_pallet
    ])