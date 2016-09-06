# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Prox-ElasticNet',
    version='0.0.1',
    description='A Python package which implements the elastic net using proximal methods',
    long_description=readme,
    author='Neil G. Marchant',
    author_email='ngmarchant@gmail.com',
    url='https://github.com/ngmarchant/prox-elasticnet',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
