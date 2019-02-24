""" Package setup """
from setuptools import setup

__version__ = '0.1'


setup(
    name='blackjack-rl',
    version=__version__,
    packages=['blackjack'],
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)
