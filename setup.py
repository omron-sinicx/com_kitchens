#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="com_kitchens",
    version="1.0.0",
    description="COM Kitchens",
    author="The COM Kitchens team",
    author_email="com_kitchens@sinicx.com",
    url="https://github.com/omron-sinicx/com_kitchens",
    install_requires=[
        "hydra-core==1.3.2",
        "lightning==2.0.5",
        "hydra-colorlog==1.2.0",
        "ffmpeg==1.4",
        "ffmpeg-python==0.2.0",
        "pyrootutils==1.0.4",
        "transformers==4.31.0",
        "datasets==2.14.3",
        "Pillow==10.0.0",
        "torch==1.12.1+cu113",
        "torchvision==0.13.1+cu113",
        "fsspec==2023.6.0",
        "numpy==1.26.4",
    ],
    extras_require={
        "develop": [
            "pre-commit",
            "jupyter",
            "ipywidgets",
        ]},
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = com_kitchens.train:main",
            "eval_command = com_kitchens.eval:main",
        ]
    },
)
