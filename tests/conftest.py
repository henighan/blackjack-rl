""" Pytest conftest """
from pytest import fixture

from blackjack.learners import BaseLearner
from blackjack.learners import MonteCarlo


@fixture()
def base_learner():
    return BaseLearner()

@fixture()
def mc_learner():
    return MonteCarlo()
