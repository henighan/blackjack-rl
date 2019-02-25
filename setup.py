""" Package setup """
from setuptools import setup

__version__ = '1.0'


setup(
    name='blackjack-rl',
    version=__version__,
    packages=['blackjack'],
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)
