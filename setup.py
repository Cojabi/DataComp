#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup.py for DataComp."""

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datacomp",
    version="0.0.1",
    author="Colin Birkenbihl",
    author_email="colin.birken@gmail.com",
    description="A small example package",
    long_description=long_description,
    url="https://github.com/Cojabi/datacomp",
    packages=setuptools.find_packages("datacomp"),
    zip_safe=False,
    keywords=[
        'clinical trial',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache Software License',
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
