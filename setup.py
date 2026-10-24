from setuptools import find_packages, setup

package_name = 'puzzlebot_line_follower'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Roberd Meza',
    maintainer_email='meza.sainz.roberd.otoniel@gmail.com',
    description='Autonomous line following for the Puzzlebot with fuzzy control, YOLOv8 and classical vision',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'line_follower_memoria = puzzlebot_line_follower.line_follower_memoria:main',
            'zebra_detector = puzzlebot_line_follower.zebra_detector:main',
            'deteccion = puzzlebot_line_follower.deteccion:main',
            'line_follower_controller = puzzlebot_line_follower.line_follower_controller:main',
            'mamdani_v1 = puzzlebot_line_follower.mamdani_v1:main',
            'sugeno_v1 = puzzlebot_line_follower.sugeno_v1:main',
        ],
    },
)
