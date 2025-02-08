import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        app_package_path = get_package_share_directory('app')
        example_package_path = get_package_share_directory('example')
    else:
        app_package_path = '/home/enrique/repos/ros2_ws/src/app'
        example_package_path = '/home/enrique/repos/ros2_ws/src/example'

 
    object_tracking_node = Node(
        package='app',
        executable='object_tracking',
        output='screen',
    )
    
    color_detect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(example_package_path, 'example/opencv/color_detect_node.launch.py')),
        # launch_arguments={
            # 'enable_display': enable_display
        # }.items()
    )

    color_tracker_node = Node(
        package='app',
        executable='color_tracker',
        output='screen',
    )
    return [
            # color_detect_launch,
            object_tracking_node,
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
