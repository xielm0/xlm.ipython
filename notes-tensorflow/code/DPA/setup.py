#!/usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'xieliming'

# setup.py文件用于构建该项目的代码和生成可执行的bin脚本，主要需要调用setuptools.setup这个函数
from setuptools import find_packages, setup


def get_long_description():
    with open('README.rst', 'rb') as reader:
        return str(reader.read())

entry_points = '''
[console_scripts]
train.bin = __train__: main
apply.bin = __apply__:main
'''

setup(
    name="dpa",
    version="1.0.0",
    description=__doc__,
    long_description=get_long_description(),
    author=__author__,
    author_email="xieliming@jd.com",
    url="http://cf.jd.com/display/sbdg/",
    license="JD",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: JD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=[
        'adsz-data',
    ],
    entry_points=entry_points,
)


