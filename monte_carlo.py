import time
import numpy as np
import common


def play_agent_hand(agent_cards, dealer_up_card, Q, deck, epsilon=0.1):
    agent_cards, agent_hand, deck= common.make_obvious_hits(agent_cards, deck)
    agent_state_action_pairs = []
    action = 'NONE'
    while (action != 'S') and (agent_hand[1] <= 21):
        agent_state_index, action_index, action = \
            common.sample_agent_state_action(
                agent_cards, dealer_up_card, Q, epsilon=epsilon)
        agent_state_action_pairs.append((agent_state_index, action_index))
        if action == 'H':
            agent_cards.append(deck.pop())
            agent_hand = common.cards_to_hand(agent_cards)
    return agent_hand, agent_state_action_pairs, deck


def play_episode(deck, Q, epsilon=0.1):
    agent_cards, dealer_up_card, dealer_down_card, deck = common.deal(deck)
    dealer_cards = [dealer_up_card, dealer_down_card]
    agent_hand, agent_state_action_pairs, deck = play_agent_hand(
        agent_cards, dealer_up_card, Q, deck, epsilon=epsilon)
    dealer_hand, deck = common.play_dealer_hand(dealer_cards, deck)
    reward = common.evaluate_reward(agent_hand, dealer_hand)
    return agent_state_action_pairs, reward, deck


def update_Q_and_counter(
        Q, counter, agent_state_action_pairs, reward, gamma=0.9):
    for ii, agent_state_action_pair in enumerate(
            reversed(agent_state_action_pairs)):
        agent_state, action = agent_state_action_pair
        G = reward*gamma**ii
        counter[agent_state][action] += 1
        Q[agent_state][action] += (
            G - Q[agent_state][action])/counter[agent_state][action]
    return Q, counter


def train(n_episodes, gamma=0.9, epsilon=0.1):
    toc = time.time()
    Q = 0.1*np.random.rand(len(common.AGENT_HANDS), 10, len(common.ACTIONS))
    counter = np.zeros_like(Q)
    deck = common.initialize_deck()
    for ii in range(n_episodes):
        if len(deck) < 16:
            deck = common.initialize_deck()
        agent_state_action_pairs, reward, deck = play_episode(
            deck, Q, epsilon=epsilon)
        Q, counter = update_Q_and_counter(
            Q, counter, agent_state_action_pairs, reward, gamma=gamma)
    common.print_stragegy_card(Q)
    print('{} episodes took {} seconds'.format(
        n_episodes, (time.time() - toc)))


if __name__=='__main__':
    train(n_episodes=int(1e5), gamma=0.9, epsilon=0.2)
