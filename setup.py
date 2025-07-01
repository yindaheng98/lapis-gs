#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

packages = ['lapisgs'] + ["lapisgs." + package for package in find_packages(where="lapisgs")]
packages += ['lapisgs.trainer.extensions.' + package for package in find_packages(where="lapisgs/trainer/extensions")]

setup(
    name="lapisgs",
    version='1.1.0',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/lapis-gs',
    description=u'Refactored python training and inference code for 3D Gaussian Splatting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages,
    package_dir={
        'lapisgs': 'lapisgs',
    },
    install_requires=[
        'gaussian-splatting',
        'reduced-3dgs'
    ]
)
