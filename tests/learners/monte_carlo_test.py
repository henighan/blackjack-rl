""" Tests for Monte Carlo learner """
import numpy as np

from blackjack.learners import MonteCarlo


def test_update_Q_episode_not_over(mc_learner):
    """ Test the Q update when the episode is still ongoing """
    mc_learner.update_Q(
        agent_state_index='state_0', action_index='action_0')
    mc_learner.update_Q(
        agent_state_index='state_1', action_index='action_1')
    assert mc_learner.counter.sum() == 0
    assert mc_learner.episode_state_action_pairs == [
        ('state_0', 'action_0'),
        ('state_1', 'action_1')]


def test_update_Q_episode_end_update_counter(mc_learner):
    """ test that when updating Q, we increment the counter, which
    counter how often we've visited each state-action pair """
    mc_learner.episode_state_action_pairs = [
        ((0, 0),  1), ((1, 1), 0), ((1, 0), 1)]
    mc_learner.update_Q(
        agent_state_index=None, action_index=None)
    assert mc_learner.counter[0, 0, 1] == 1
    assert mc_learner.counter[1, 1, 0] == 1
    assert mc_learner.counter[1, 0, 1] == 1
    assert mc_learner.counter[0, 0, 0] == 0


def test_update_Q_episode_end_update_Q(mc_learner):
    """ test that Q is properly updated, with discounting """
    mc_learner.episode_state_action_pairs = [
        ((0, 0),  1), ((1, 1), 0), ((1, 0), 1)]
    reward = 1
    mc_learner.Q = np.zeros_like(mc_learner.Q)
    mc_learner.update_Q(
        agent_state_index=None, action_index=None, reward=reward)
    assert mc_learner.Q[0, 0, 0] == 0
    assert mc_learner.Q[1, 0, 1] == 1
    assert mc_learner.Q[1, 1, 0] == 1*mc_learner.gamma
    assert mc_learner.Q[0, 0, 1] == 1*mc_learner.gamma**2
    assert mc_learner.episode_state_action_pairs == []
