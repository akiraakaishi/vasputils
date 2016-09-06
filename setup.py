import os
from setuptools import setup

import vasputils

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='vasputils',
    version='.'.join(map(str, vasputils.VERSION)),
    packages=['vasputils'],
    entry_points={
        'console_scripts': [
            'chg_split = vasputils.commands:chg_split',
            'vasprun = vasputils.commands:vasprun',
        ],
    },
    include_package_data=False,
    license='MIT License',
    description='Utility tools for VASP',
    long_description=README,
    author='Akira Akaishi',
    author_email='akira.akaishi@gmail.com',
    classifiers=[
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
)
