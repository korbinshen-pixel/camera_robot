#!/usr/bin/env python3
"""
pallet_pickup.launch.py
=======================
基于 complete.launch.py + navigation.launch.py 整合的托盘自动拾取启动文件

启动内容：
  1. Gazebo + 机器人 + 托盘（通过 spawn_robot.launch.py）
  2. 静态 TF（base_link / base_footprint / lidar_link / camera_link）
  3. Nav2（map server + AMCL + planner/controller）
  4. RViz
  5. /cmd_vel -> /camera_robot/cmd_vel relay
  6. deep_pallet_detector
  7. pallet_pickup_mission

用法：
  ros2 launch camera_robot pallet_pickup.launch.py
  ros2 launch camera_robot pallet_pickup.launch.py \
      map:=/home/skj/camera_robot_ws/maps/warehouse_map.yaml \
      model_path:=/home/skj/camera_robot_ws/src/camera_robot/camera_robot/resource/weights/best_model_phi0.pth
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('camera_robot')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    default_map = os.path.join(
        os.path.expanduser('~'),
        'camera_robot_ws', 'maps', 'warehouse_map.yaml'
    )
    default_nav2_params = os.path.join(pkg, 'config', 'nav2_params.yaml')
    default_detector_model = os.path.join(
        os.path.expanduser('~'),
        'camera_robot_ws', 'src', 'camera_robot', 'camera_robot',
        'resource', 'weights', 'best_model_phi0.pth'
    )
    default_rviz = os.path.join(nav2_pkg, 'rviz', 'nav2_default_view.rviz')

    # ── Launch 参数 ───────────────────────────────────────────
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map,
        description='已保存地图的 yaml 路径'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='是否使用仿真时间'
    )

    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params',
        default_value=default_nav2_params,
        description='Nav2 参数文件路径'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_detector_model,
        description='deep_pallet_detector 模型权重路径'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='推理设备: cuda / cpu'
    )

    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value='true',
        description='是否使用深度信息'
    )

    goal_offset_arg = DeclareLaunchArgument(
        'goal_offset_m',
        default_value='0.6',
        description='导航到托盘正前方的偏移距离'
    )

    lock_confirm_arg = DeclareLaunchArgument(
        'lock_confirm_count',
        default_value='5',
        description='连续检测多少帧后锁定托盘'
    )

    nav_timeout_arg = DeclareLaunchArgument(
        'nav_timeout_sec',
        default_value='60.0',
        description='导航超时时间（秒）'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='是否启动 RViz'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_cfg = LaunchConfiguration('map')
    nav2_params_cfg = LaunchConfiguration('nav2_params')
    model_path_cfg = LaunchConfiguration('model_path')
    device_cfg = LaunchConfiguration('device')
    use_depth_cfg = LaunchConfiguration('use_depth')
    goal_offset_cfg = LaunchConfiguration('goal_offset_m')
    lock_confirm_cfg = LaunchConfiguration('lock_confirm_count')
    nav_timeout_cfg = LaunchConfiguration('nav_timeout_sec')
    rviz_cfg = LaunchConfiguration('rviz')

    # ── 1. 启动 Gazebo + 生成机器人 + 托盘 ─────────────────────
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'spawn_robot.launch.py')
        )
    )

    # ── 2. 静态 TF ────────────────────────────────────────────
    # navigation.launch.py 里原本是 base_link -> base_footprint
    tf_base_to_footprint = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_footprint',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'base_footprint',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_lidar',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0.23',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'lidar_link',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=[
            '--x', '0.25',
            '--y', '0',
            '--z', '0.28',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # ── 3. Nav2 ───────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_cfg,
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_cfg,
            'autostart': 'true',
        }.items()
    )

    # ── 4. RViz ───────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz_cfg),
        output='screen'
    )

    # ── 5. cmd_vel relay ──────────────────────────────────────
    # Nav2 默认发 /cmd_vel，但你的小车监听 /camera_robot/cmd_vel
    cmd_vel_relay = Node(
        package='topic_tools',
        executable='relay',
        name='cmd_vel_relay',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['/cmd_vel', '/camera_robot/cmd_vel'],
        output='screen'
    )

    # ── 6. deep_pallet_detector ───────────────────────────────
    deep_pallet_detector = Node(
        package='camera_robot',
        executable='deep_pallet_detector',
        name='deep_pallet_detector',
        parameters=[{
            'use_sim_time': use_sim_time,
            'model_path': model_path_cfg,
            'device': device_cfg,
            'use_depth': PythonExpression(["'", use_depth_cfg, "' == 'true'"]),
        }],
        output='screen',
        emulate_tty=True,
    )

    # ── 7. pallet_pickup_mission ──────────────────────────────
    pallet_pickup_mission = Node(
        package='camera_robot',
        executable='pallet_pickup_mission',
        name='pallet_pickup_mission',
        parameters=[{
            'use_sim_time': use_sim_time,
            'goal_offset_m': goal_offset_cfg,
            'lock_confirm_count': lock_confirm_cfg,
            'nav_timeout_sec': nav_timeout_cfg,
        }],
        output='screen',
        emulate_tty=True,
    )

    # ── 启动时序 ──────────────────────────────────────────────
    nav2_delayed = TimerAction(period=5.0, actions=[nav2])
    rviz_delayed = TimerAction(period=6.0, actions=[rviz])
    relay_delayed = TimerAction(period=6.5, actions=[cmd_vel_relay])
    detector_delayed = TimerAction(period=8.0, actions=[deep_pallet_detector])
    mission_delayed = TimerAction(period=12.0, actions=[pallet_pickup_mission])

    return LaunchDescription([
        map_arg,
        use_sim_time_arg,
        nav2_params_arg,
        model_path_arg,
        device_arg,
        use_depth_arg,
        goal_offset_arg,
        lock_confirm_arg,
        nav_timeout_arg,
        rviz_arg,

        LogInfo(msg='[1/6] 启动 Gazebo + 机器人 + 托盘...'),
        spawn,

        LogInfo(msg='[2/6] 发布静态 TF...'),
        tf_base_to_footprint,
        tf_base_to_lidar,
        tf_base_to_camera,

        LogInfo(msg='[3/6] Nav2 将在 5s 后启动...'),
        nav2_delayed,

        LogInfo(msg='[4/6] RViz 将在 6s 后启动...'),
        rviz_delayed,

        LogInfo(msg='[5/6] cmd_vel relay 将在 6.5s 后启动...'),
        relay_delayed,

        LogInfo(msg='[6/6] detector / mission 将依次延迟启动...'),
        detector_delayed,
        mission_delayed,
    ])