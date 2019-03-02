""" tests for sarsa learner """
import numpy as np

from blackjack.learners import Sarsa


def test_update_Q_update_last_state_action(sarsa_learner):
    """ Test that update_Q updates the last state-action pair """
    assert sarsa_learner.last_state_index is None
    assert sarsa_learner.last_action_index is None
    state_index = (0, 0)
    action_index = 0
    sarsa_learner.update_Q(state_index, action_index)
    assert sarsa_learner.last_state_index == state_index
    assert sarsa_learner.last_action_index == action_index


def test_update_Q_update_reward_update(sarsa_learner):
    """ Test that update_Q updates correctly based on the reward """
    state_index = (0, 0)
    action_index = 0
    reward = 1
    sarsa_learner.last_state_index = state_index
    sarsa_learner.last_action_index = action_index
    sarsa_learner.Q = np.zeros_like(sarsa_learner.Q)
    sarsa_learner.update_Q(None, None, reward=reward)
    assert sarsa_learner.last_state_index is None
    assert sarsa_learner.last_action_index is None
    assert (sarsa_learner.Q[state_index][action_index]
            == sarsa_learner.alpha*reward)
    assert sarsa_learner.Q.sum() == sarsa_learner.alpha*reward


def test_update_Q_update_next_state_update(sarsa_learner):
    """ Test that update_Q updates correctly based on next state-action value
    """
    state_index = (0, 0)
    action_index = 0
    next_state_index = (1, 1)
    next_action_index = 1
    sarsa_learner.Q[state_index][action_index] = 0
    sarsa_learner.Q[next_state_index][next_action_index] = 1
    sarsa_learner.last_state_index = state_index
    sarsa_learner.last_action_index = action_index
    sarsa_learner.update_Q(next_state_index, next_action_index, reward=0)
    assert (sarsa_learner.Q[state_index][action_index]
            == sarsa_learner.alpha*sarsa_learner.gamma)
