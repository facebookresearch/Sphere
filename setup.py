# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="sphere",
    version="0.0.1",
    description="The Sphere library",
    url="",  # TODO
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: CC-BY-NC",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "black",
        "requests",
        "tqdm",
    ],
)
