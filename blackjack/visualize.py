""" Visualizations for learners """
import numpy as np

from matplotlib import pyplot as plt
from blackjack.common import ACTIONS, AGENT_HANDS


def plot_eval_reward(*args):
    """ Plot the estimated reward from evaluations with error bars """
    for learner in args:
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


def plot_strategy_card(learner):
    """ Make a heatmap plotting the strategy card according to optimal
    strategy obtained from learner.Q, the action-value function """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots()
    Q_argmax = np.argmax(learner.Q, axis=-1)
    ax.imshow(Q_argmax, cmap='coolwarm')
    ylabels = [' '.join(map(str, hand)) for hand in AGENT_HANDS]
    xlabels = list(map(str, range(2, 11))) + ['A']
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    ax.set_xticks(np.arange(len(xlabels)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(ylabels)+1)-.5, minor=True)
    ax.grid(which="minor", linestyle='-', color='k', linewidth=1)
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            ax.text(j, i, ACTIONS[Q_argmax[i, j]],
                    ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()
