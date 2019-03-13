#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup.py for DataComp."""

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

PACKAGES = setuptools.find_packages(where='src')

setuptools.setup(
    name="datacomp",
    version='0.0.3',
    author="Colin Birkenbihl",
    author_email="colin.birken@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cojabi/datacomp",
    packages=PACKAGES,
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas',
        'xlrd',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib_venn',
        'pymatch',
        'matplotlib',
        'statsmodels',
        'seaborn',
    ]
)
