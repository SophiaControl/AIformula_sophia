from setuptools import setup

package_name = 'obsticle_avoidence'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='zw01',
    maintainer_email='zw01@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'b_spline = obsticle_avoidence.b_spline:main',
        ],
    },
)
