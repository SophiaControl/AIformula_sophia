from setuptools import setup

package_name = 'trajectory_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	     'lya_follower_connected_omegat_global=trajectory_follower.lya_follower_connected_omegat_global:main',
             'lya_record=trajectory_follower.lya_record:main',
             'lya_oa=trajectory_follower.lya_oa:main',
             'lya_follower_fixedpath_record=trajectory_follower.lya_follower_fixedpath_record:main',
             'lya_baseline_follower_fixedpath_record=trajectory_follower.lya_baseline_follower_fixedpath_record:main',
        ],
    },
)
