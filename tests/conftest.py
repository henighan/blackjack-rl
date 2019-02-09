""" Pytest conftest """
from pytest import fixture

from blackjack.learners import BaseLearner


@fixture()
def base_learner():
    return BaseLearner()
