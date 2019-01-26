"""
Test sarsa training methods
"""
import sarsa
import numpy as np


def test_sarsa_update_reward():
    Q = np.zeros([1, 1, 1])
    agent_state_index, action_index = (0, 0), 0
    reward = 1
    alpha = 0.5
    ret = sarsa.update(
        Q, agent_state_index, action_index, reward=reward, alpha=alpha)
    np.testing.assert_array_equal(ret, 0.5*np.ones([1, 1, 1]))


def test_sarsa_update_next_state_zero():
    Q = np.zeros([1, 1, 1])
    agent_state_index, action_index = (0, 0), 0
    next_agent_state_index, next_action_index = (0, 0), 0
    alpha = 0.5
    gamma = 0.9
    ret = sarsa.update(
        Q, agent_state_index, action_index,
        next_agent_state_index=next_agent_state_index,
        next_action_index=next_action_index, alpha=alpha, gamma=gamma)
    np.testing.assert_array_equal(ret, np.zeros([1, 1, 1]))


def test_sarsa_update_next_state_nonzero():
    Q = np.zeros([2, 2, 2])
    agent_state_index, action_index = (0, 0), 0
    next_agent_state_index, next_action_index = (1, 1), 1
    Q[next_agent_state_index][next_action_index] = 0.3
    alpha = 0.5
    gamma = 0.9
    ret = sarsa.update(
        Q, agent_state_index, action_index,
        next_agent_state_index=next_agent_state_index,
        next_action_index=next_action_index, alpha=alpha, gamma=gamma)
    assert ret[agent_state_index][action_index] == 0.3*alpha*gamma
    assert ret.sum() == 0.3*alpha*gamma + 0.3


def test_sarsa_update_same_next_state_nonzero():
    Q = np.zeros([1, 1, 1])
    agent_state_index, action_index = (0, 0), 0
    next_agent_state_index, next_action_index = (0, 0), 0
    Q[next_agent_state_index][next_action_index] = 0.3
    alpha = 0.5
    gamma = 0.9
    ret = sarsa.update(
        Q, agent_state_index, action_index,
        next_agent_state_index=next_agent_state_index,
        next_action_index=next_action_index, alpha=alpha, gamma=gamma)
    updated_Q = 0.3 + (gamma-1)*alpha*0.3*np.ones([1, 1, 1])
    print(updated_Q)
    print(ret)
    np.testing.assert_array_almost_equal(ret, updated_Q)
