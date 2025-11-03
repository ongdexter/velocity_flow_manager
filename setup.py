from setuptools import setup
import os
from glob import glob

package_name = 'velocity_flow_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'PyQt5'],
    python_requires='>=3.8',
    zip_safe=True,
    maintainer='Dexter Ong',
    maintainer_email='dexterong94@gmail.com',
    description='Manage 2D waypoints from RViz and publish nav_msgs/Path',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'velocity_flow_gui = velocity_flow_manager.velocity_flow_gui:main',
    ]},
)
