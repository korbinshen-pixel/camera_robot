import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg      = get_package_share_directory('camera_robot')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(
            os.path.expanduser('~'),
            'camera_robot_ws', 'maps', 'warehouse_map.yaml'
        ),
        description='已保存地图的 yaml 路径'
    )

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'spawn_robot.launch.py')
        )
    )

    # base_link → base_footprint（让 nav2 默认的 base_footprint 能找到）
    tf_base_to_footprint = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_footprint',
        arguments=['0', '0', '0', '0', '0', '0',
                   'base_link', 'base_footprint'],
        parameters=[{'use_sim_time': True}]
    )

    # base_link → lidar_link
    tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_lidar',
        arguments=['0', '0', '0.23', '0', '0', '0',
                   'base_link', 'lidar_link'],
        parameters=[{'use_sim_time': True}]
    )

    # base_link → camera_link
    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=['0.25', '0', '0.28', '0', '0', '0',
                   'base_link', 'camera_link'],
        parameters=[{'use_sim_time': True}]
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map':          LaunchConfiguration('map'),
            'use_sim_time': 'true',
            'params_file':  os.path.join(pkg, 'config', 'nav2_params.yaml'),
            'autostart':    'true',
        }.items()
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d',
                   os.path.join(nav2_pkg, 'rviz', 'nav2_default_view.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # 将 Nav2 输出的 /cmd_vel 转发到小车实际监听的话题
    cmd_vel_relay = Node(
        package='topic_tools',
        executable='relay',
        name='cmd_vel_relay',
        parameters=[{'use_sim_time': True}],
        arguments=['/cmd_vel', '/camera_robot/cmd_vel'],
    )

    return LaunchDescription([
        map_arg,
        spawn,
        tf_base_to_footprint,   # ← 关键：连接两棵 TF 树
        tf_base_to_lidar,
        tf_base_to_camera,
        nav2,
        rviz,
        cmd_vel_relay,
    ])