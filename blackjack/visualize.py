""" Visualizations for learners """
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
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
    fig, axes = plt.subplots(nrows=1, ncols=2)
    Q_argmax = np.argmax(learner.Q, axis=-1)
    ylabels = [' '.join(map(str, hand)) for hand in AGENT_HANDS]
    xlabels = list(map(str, range(2, 11))) + ['A']
    annot = np.array(ACTIONS)[Q_argmax]
    ylabels = [' '.join(map(str, hand)) for hand in AGENT_HANDS]
    xlabels = list(map(str, range(2, 11))) + ['A']
    sns.heatmap(Q_argmax[:10], ax=axes[0], square=True, cbar=False,
                cmap='coolwarm', linewidth=0.05, annot=annot[:10], fmt='s',
                xticklabels=xlabels, yticklabels=ylabels[:10])
    sns.heatmap(Q_argmax[-10:], ax=axes[1], square=True, cbar=False,
                cmap='coolwarm', linewidth=0.05, annot=annot[-10:], fmt='s',
                xticklabels=xlabels, yticklabels=ylabels[-10:])
    fig.tight_layout()
    return fig, axes
