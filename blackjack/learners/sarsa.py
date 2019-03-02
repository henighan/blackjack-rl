""" Sarsa Learner """
from blackjack.learners import BaseLearner


class Sarsa(BaseLearner):
    """ Sarsa learner """

    def __init__(self, epsilon=0.1, window_size=1000, name=None, gamma=0.9,
                 alpha=0.05):
        super().__init__(epsilon=epsilon, window_size=window_size, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.last_state_index = None
        self.last_action_index = None

    def update_Q(self, agent_state_index, action_index, reward=0):
        """ update the action-value function """
        if self.last_action_index is None:
            self.last_state_index = agent_state_index
            self.last_action_index = action_index
            return self
        estimated_return = reward
        if agent_state_index:
            estimated_return += self.gamma*self.Q[
                agent_state_index][action_index]
        update = self.alpha*(
            estimated_return
            - self.Q[self.last_state_index][self.last_action_index])
        self.Q[self.last_state_index][self.last_action_index] += update
        self.last_state_index = agent_state_index
        self.last_action_index = action_index
        return self
