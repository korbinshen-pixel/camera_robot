from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('camera_robot')

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'spawn_robot.launch.py')
        )
    )

    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            os.path.join(pkg, 'config', 'slam_params.yaml'),
            {'use_sim_time': True}
        ]
    )

    teleop = Node(
        package='camera_robot',
        executable='robot_controller',
        name='robot_controller',
        prefix='xterm -e',
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    # ── 静态 TF：base_link → lidar_link ──
    # 与 model.sdf 中 lidar_link 的 pose 一致：(0 0 0.23)
    tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_lidar',
        arguments=['0', '0', '0.23', '0', '0', '0',
                   'base_link', 'lidar_link']
    )

    # ── 静态 TF：base_link → camera_link ──
    # 与 model.sdf 中 camera_link 的 pose 一致：(0.25 0 0.28)
    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=['0.25', '0', '0.28', '0', '0', '0',
                   'base_link', 'camera_link']
    )

    # ── robot_state_publisher：发布 odom → base_link ──
    # diff_drive 插件已经在发布这个，但加一个 robot_state_publisher
    # 确保 TF 树完整
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # base_footprint → base_link（单位变换，完全重合）
    tf_footprint_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_footprint_to_base',
        arguments=['0', '0', '0', '0', '0', '0',
                'base_footprint', 'base_link'],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        spawn,
        tf_base_to_lidar,
        tf_base_to_camera,
        robot_state_pub,
        slam,
        teleop,
        rviz,
        tf_footprint_to_base,
    ])