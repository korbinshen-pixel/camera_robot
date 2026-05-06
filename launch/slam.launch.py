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

    # base_link -> lidar_link
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
            '--child-frame-id', 'lidar_link'
        ],
        parameters=[{'use_sim_time': True}]
    )

    # base_link -> camera_link
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
            '--child-frame-id', 'camera_link'
        ],
        parameters=[{'use_sim_time': True}]
    )

    # base_footprint -> base_link
    tf_footprint_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_footprint_to_base',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link'
        ],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        spawn,
        tf_base_to_lidar,
        tf_base_to_camera,
        tf_footprint_to_base,
        slam,
        teleop,
        rviz,
    ])