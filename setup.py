#!/usr/bin/env python

from setuptools import setup

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""

setup(
    name="ensalada",
    version="0.0.1",
    author="James Priestley, Zhenrui Liao",
    author_email="zhenrui.liao@columbia.edu, jbp2150@columbia.edu",
    description=("Supervised latent Dirichlet allocation for neural data analysis"),
    license="GNU GPLv2",
    keywords="supervised topic model neural ensemble",
    packages=['ensalada'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=[],
)
