from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='neurofly1',
        description='Robot namespace'
    )

    publish_topic_arg = DeclareLaunchArgument('publish_topic',
                                              default_value='trackers_manager/velocity_tracker/goal',
                                              description='Velocity publish topic')

    publish_topic = LaunchConfiguration('publish_topic')

    return LaunchDescription([
        robot_arg,
        publish_topic_arg,
        Node(
            package='velocity_flow_manager',
            executable='velocity_flow_gui',
            name='velocity_flow_gui',
            namespace=LaunchConfiguration('robot'),
            output='screen',
            parameters=[{'publish_topic': publish_topic}]
        )
    ])
