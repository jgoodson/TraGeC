# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


def get_version():
    directory = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(directory, 'tragec', '__init__.py')
    with open(init_file) as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        raise RuntimeError("Unable to find version string.")


with open('README.md', 'r') as rf:
    README = rf.read()

with open('TAPE-LICENSE', 'r') as lf:
    LICENSE = lf.read()

with open('requirements.txt', 'r') as reqs:
    requirements = reqs.read().split()

setup(
    name='tragec',
    packages=find_packages(),
    version=get_version(),
    description="Deep Learning for Gene Cluster Analysis",
    author="Jonathan Goodson",
    author_email='jgoodson@umd.edu',
    url='https://github.com/jgoodson/dlgec',
    license=LICENSE,
    keywords=['Proteins', 'Deep Learning', 'Pytorch', 'Gene Clusters'],
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'tragec-train = tragec.main:run_train',
            # 'tragec-train-distributed = tragec.main:run_train_distributed',
            'tragec-eval = tragec.main:run_eval',
            #'tragec-embed = tragec.main:run_embed',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
