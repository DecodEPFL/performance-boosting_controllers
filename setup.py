#!/usr/bin/env python
import setuptools


setuptools.setup(
    name='performance-boosting_controllers',
    version='1.0',
    url='https://github.com/DecodEPFL/performance-boosting_controllers',
    license='CC-BY-4.0 License',
    author='Clara Galimberti',
    author_email='clara.galimberti@epfl.ch',
    description='Learning to boost the performance of stable nonlinear closed-loop systems',
    packages=setuptools.find_packages(),
    install_requires=['torch>=2.2.0',
                      'numpy>=1.26.4',
                      'matplotlib==3.6.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.10',
)
