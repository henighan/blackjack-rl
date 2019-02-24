""" Monte Carlo Learner """
import numpy as np

from blackjack.learners import BaseLearner


class MonteCarlo(BaseLearner):
    """ Monte Carlo learner """

    def __init__(self, epsilon=0.1, window_size=1000, gamma=0.9):
        super().__init__(epsilon=epsilon, window_size=window_size)
        self.gamma = gamma
        self.episode_state_action_pairs = []
        self.counter = np.zeros_like(self.Q)

    def update_Q(self, agent_state_index, action_index, reward=0):
        """ update the action-value function """
        if agent_state_index is None:
            # have reached the end of the episode
            # update Q for all visited state-action pairs
            for step, state_action_pair in enumerate(
                    reversed(self.episode_state_action_pairs)):
                state, action = state_action_pair
                G = reward*self.gamma**step
                self.counter[state][action] += 1
                self.Q[state][action] += (
                    G - self.Q[state][action])/self.counter[state][action]
            # reinitialize episode state-action pairs for next episode
            self.episode_state_action_pairs = []
        else:
            # still in episode, store state-action pairs
            self.episode_state_action_pairs.append(
                (agent_state_index, action_index))
        return self


if __name__=='__main__':
    learner = MonteCarlo()
    learner.train_and_evaluate(n_episodes=int(3e4))
