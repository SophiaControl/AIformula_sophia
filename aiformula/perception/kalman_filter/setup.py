from setuptools import setup

package_name = 'kalman_filter'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Yu Narukami',
    maintainer_email='1037657394@qq.com',
    description='A ROS2 package implementing Kalman Filter for smoothing lane points and angles',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
            'kalman0225 = kalman_filter.kalman0225:main',
            'withoutkalman = kalman_filter.withoutkalman:main',
            'withoutkalman_0312 = kalman_filter.withoutkalman_0312:main',
        ],
    },
)
