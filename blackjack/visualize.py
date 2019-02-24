""" Visualizations for learners """
import numpy as np

from matplotlib import pyplot as plt
from blackjack.common import ACTIONS, AGENT_HANDS


def plot_eval_reward(learner):
    """ Plot the estimated reward from evaluations with error bars """
    n_training_episodes, mean_reward, err = zip(*learner.evaluations)
    plt.errorbar(n_training_episodes, mean_reward, yerr=err,
                 label=str(learner.name))
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Estimated Mean Reward')
    plt.legend()
    plt.gca().set_xscale('log')
    plt.show()


def print_strategy_card(learner):
    """ Given the learner, print the optimal strategy card according
    to learner.Q """
    card_actions = np.array(ACTIONS)[np.argmax(learner.Q, axis=-1)]
    print('     ' + ' '.join(map(str, range(2, 11))) + 'A')
    for agent_hand, row in zip(reversed(AGENT_HANDS), reversed(card_actions)):
        print(' '.join(map(str, agent_hand)) + ' ' + ' '.join(row))
