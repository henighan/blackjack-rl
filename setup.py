""" Package setup """
from setuptools import setup

__version__ = '1.1'


setup(
    name='blackjack-rl',
    version=__version__,
    packages=['blackjack', 'blackjack.learners'],
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
    ],
)
