""" Pytest conftest """
from pytest import fixture

from blackjack.learners import BaseLearner, MonteCarlo, Sarsa


@fixture()
def base_learner():
    return BaseLearner()

@fixture()
def mc_learner():
    return MonteCarlo()

@fixture()
def sarsa_learner():
    return Sarsa()
