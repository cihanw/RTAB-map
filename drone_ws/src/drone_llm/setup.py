import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'drone_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cihan',
    maintainer_email='cihan@todo.todo',
    description='Local-LLM natural-language commander for the drone_sim stack.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_bridge = drone_llm.llm_bridge_node:main',
            'llm_console = drone_llm.console:main',
            'llm_web = drone_llm.web_ui:main',
        ],
    },
)
