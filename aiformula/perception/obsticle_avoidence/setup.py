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
        'oa_old = obsticle_avoidence.oa_old:main',
        'oa_new = obsticle_avoidence.oa_new:main',
        'oa_neo = obsticle_avoidence.oa_neo:main',
        'oa_neov1.2 = obsticle_avoidence.oa_neov1.2:main',
        'bspline_test_oa_1 = obsticle_avoidence.bspline_test_oa_1:main',
        'bspline_test_oa_2 = obsticle_avoidence.bspline_test_oa_2:main',
        'bspline_test_oa_3 = obsticle_avoidence.bspline_test_oa_3:main',
        'b_spline_final_1 = obsticle_avoidence.b_spline_final_1:main',
        'b_spline_final_2 = obsticle_avoidence.b_spline_final_2:main',
        'b_spline_final_3 = obsticle_avoidence.b_spline_final_3:main',
        'b_spline_final_3re = obsticle_avoidence.b_spline_final_3re:main',
        'b_spline_final_3b = obsticle_avoidence.b_spline_final_3b:main',
        'data_record = obsticle_avoidence.data_record:main',
        'b_spline_final_1a = obsticle_avoidence.b_spline_final_1a:main',
        ],
    },
)
