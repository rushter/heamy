#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'scikit-learn>=0.17.0',
    'pandas>=0.17.0',
    'six>=1.10.0',
    'scipy>=0.16.0',
    'numpy>=1.7.0',
]

test_requirements = [
    'pytest',
    'coveralls',
]

setup(
    name='heamy',
    version='0.0.6',
    description="A set of useful tools for competitive data science.",
    long_description=readme,
    author="Artem Golubin",
    author_email='me@rushter.com',
    url='https://github.com/rushter/heamy',
    packages=find_packages(exclude=['tests', ]),
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='heamy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=['pytest-runner'],
)
