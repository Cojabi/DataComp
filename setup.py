#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pandas>=0.18.1', 'numpy', 'scipy']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Colin Birkenbihl",
    author_email='colin.birken@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="compare feature distributions of datasets",
    entry_points={
        'console_scripts': [ ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n',
    include_package_data=True,
    keywords='',
    name='compare_features',
    packages=find_packages(include=['src']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Cojabi/DataComp/',
    version='0.1.0',
    zip_safe=False,
)
