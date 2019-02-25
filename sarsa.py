"""
Sarsa training methods
"""
import time
import numpy as np
import common
from collections import defaultdict


def update(Q, agent_state_index, action_index,
                 next_agent_state_index=None, next_action_index=None,
                 reward=0, alpha=0.05, gamma=0.9):
    """ Update Q given the current state-action pair,
    next state-action pair, and reward. alpha is the learning rate while
    gamma is the discount factor"""
    next_value = 0
    if next_agent_state_index:
        next_value = Q[next_agent_state_index][next_action_index]
    Q[agent_state_index][action_index] += alpha*(
        reward + gamma*next_value - Q[agent_state_index][action_index])
    return Q

def play_episode(deck, Q, alpha=0.05, gamma=0.9, epsilon=0.1):
    """
    Play out a single episode, making sarsa updates at each timestep
    """
    agent_cards, dealer_up_card, dealer_down_card, deck = common.deal(deck)
    agent_cards, agent_hand, deck = common.make_obvious_hits(agent_cards, deck)
    agent_state_index, action_index, action = common.sample_agent_state_action(
        agent_cards, dealer_up_card, Q, epsilon=epsilon)
    if action == 'H':
        agent_cards.append(deck.pop())
        agent_hand = common.cards_to_hand(agent_cards)
    while (action != 'S') and (agent_hand[1] <= 21):
        next_agent_state_index, next_action_index, next_action = \
            common.sample_agent_state_action(agent_cards, dealer_up_card,
                                             Q, epsilon=epsilon)
        Q = update(Q, agent_state_index, action_index,
                         next_agent_state_index=next_agent_state_index,
                         next_action_index=next_action_index,
                         reward=0, alpha=alpha, gamma=gamma)
        agent_state_index = next_agent_state_index
        action_index, action = next_action_index, next_action
        if action == 'H':
            agent_cards.append(deck.pop())
            agent_hand = common.cards_to_hand(agent_cards)
    dealer_cards = [dealer_up_card, dealer_down_card]
    dealer_hand, deck = common.play_dealer_hand(dealer_cards, deck)
    reward = common.evaluate_reward(agent_hand, dealer_hand)
    Q = update(Q, agent_state_index, action_index, reward=reward,
                     alpha=alpha)
    return Q, reward, deck


def train(n_episodes, alpha=0.05, gamma=0.9, epsilon=0.1,
        evaluation_episodes=None, n_evaluate_episodes=None):
    toc = time.time()
    Q = 0.1*np.random.rand(len(common.AGENT_HANDS), 10, len(common.ACTIONS))
    deck = common.initialize_deck()
    rewards = np.zeros(n_episodes)
    evaluation = defaultdict(list)
    eval_eps_copy = evaluation_episodes[:]
    eval_eps_copy.sort(reverse=True)
    eval_episode = eval_eps_copy.pop()
    for ii in range(n_episodes):
        if len(deck) < 16:
            deck = common.initialize_deck()
        Q, reward, deck = play_episode(
            deck, Q, alpha=alpha, gamma=gamma, epsilon=epsilon)
        rewards[ii] = reward
        if ii == eval_episode:
            mean_reward, err = common.evaluate_strategy(
                Q, n_episodes=n_evaluate_episodes)
            evaluation['mean_reward'].append(mean_reward)
            evaluation['95%'].append(err)
            evaluation['episode_no'].append(ii)
            if eval_eps_copy:
                eval_episode = eval_eps_copy.pop()
    print('SARSA training {} episodes took {} seconds'.format(
        n_episodes, (time.time() - toc)))
    return Q, rewards, evaluation


if __name__=='__main__':
    Q, rewards = train(n_episodes=int(1e5), alpha=0.01, gamma=0.9, epsilon=0.1)
    common.print_stragegy_card(Q)
