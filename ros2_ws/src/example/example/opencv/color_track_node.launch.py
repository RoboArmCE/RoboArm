
import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    display = LaunchConfiguration('display', default='true')
    display_arg = DeclareLaunchArgument('display', default_value=display)
    start = LaunchConfiguration('start', default='true')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    if compiled == 'True':
        sdk_package_path = get_package_share_directory('sdk')
        peripherals_package_path = get_package_share_directory('peripherals')
        kinematics_package_path = get_package_share_directory('kinematics')
        example_package_path = get_package_share_directory('example')
    else:
        sdk_package_path = '/home/jetson/repos/RoboArm/ros2_ws/src/driver/sdk'
        peripherals_package_path = '/home/jetson/repos/RoboArm/ros2_ws/src/peripherals'
        kinematics_package_path = '/home/jetson/repos/RoboArm/ros2_ws/src/driver/kinematics'
        example_package_path = '/home/jetson/repos/RoboArm/ros2_ws/src/example'

    sdk_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sdk_package_path, 'launch/jetarm_sdk.launch.py')),
    )

    kinematics_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kinematics_package_path, 'launch/kinematics_node.launch.py')),
    )

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )


    color_track_node = Node(
        package='app',
        executable='object_tracking',
        output='screen',
        parameters=[{ 'start': start, 'display': display}]
    )

    return [start_arg,
            display_arg,
            sdk_launch,
            kinematics_launch,
            depth_camera_launch,
            color_track_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

