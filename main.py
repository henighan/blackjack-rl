import time
from random import shuffle
import numpy as np
import cProfile

ACTIONS = ['S', 'H'] # S: stay, H: hit
Q = 0.1*np.random.rand(21 - 4 + 1, 9, len(ACTIONS))
counter = np.zeros_like(Q)


def initialize_deck():
    deck = 4*([2, 3, 4, 5, 6, 7, 8, 9] + 4*[10])
    shuffle(deck)
    return deck


def agent_hand_to_index(agent_hand):
    return sum(agent_hand) - 4

def dealer_up_card_to_index(dealer_up_card):
    return dealer_up_card - 2

def agent_state_to_index(agent_state):
    agent_hand, dealer_up_card = agent_state
    return agent_hand_to_index(agent_hand), dealer_up_card_to_index(dealer_up_card)


def deal(deck):
    agent_hand = [deck.pop()]
    dealer_up_card = deck.pop()
    agent_hand.append(deck.pop())
    dealer_down_card = deck.pop()
    return agent_hand, dealer_up_card, dealer_down_card, deck


def play_agent_hand(agent_hand, dealer_up_card, Q, deck, epsilon=0.1):
    agent_state_action_pairs = []
    action = 'H'
    while (action != 'S') & (sum(agent_hand) <= 21):
        agent_state = (agent_hand, dealer_up_card)
        agent_state_index = agent_state_to_index(agent_state)
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, Q.shape[-1])
        else:
            action_index = np.argmax(Q[agent_state_index])
        agent_state_action_pairs.append((agent_state_index, action_index))
        action = ACTIONS[action_index]
        if action == 'H':
            agent_hand.append(deck.pop())
    return agent_hand, agent_state_action_pairs, deck


def play_dealer_hand(dealer_hand, deck):
    while sum(dealer_hand) < 17:
        dealer_hand.append(deck.pop())
    return dealer_hand, deck


def evaluate_reward(agent_hand, dealer_hand):
    agent_sum, dealer_sum = sum(agent_hand), sum(dealer_hand)
    if agent_sum > 21:
        return -1
    elif dealer_sum > 21:
        return 1
    elif agent_sum == dealer_sum:
        return 0
    elif agent_sum > dealer_sum:
        return 1
    return -1


def play_episode(deck, Q, epsilon=0.1):
    agent_hand, dealer_up_card, dealer_down_card, deck = deal(deck)
    dealer_hand = [dealer_up_card, dealer_down_card]
    agent_hand, agent_state_action_pairs, deck = play_agent_hand(
        agent_hand, dealer_up_card, Q, deck, epsilon=epsilon)
    dealer_hand, deck = play_dealer_hand(dealer_hand, deck)
    reward = evaluate_reward(agent_hand, dealer_hand)
    return agent_state_action_pairs, reward, deck


def update_Q_and_counter(
        Q, counter, agent_state_action_pairs, reward, gamma=0.9):
    for ii, agent_state_action_pair in enumerate(
            reversed(agent_state_action_pairs)):
        agent_state, action = agent_state_action_pair
        G = reward*gamma**ii
        counter[agent_state][action] += 1
        Q[agent_state][action] += (G - Q[agent_state][action])/counter[agent_state][action]
    return Q, counter


def print_stragegy_card(Q):
    card_actions = np.array(ACTIONS)[np.argmax(Q, axis=-1)]
    for ii, row in enumerate(card_actions):
        agent_hand_display = str(4+ii) 
        if ii+4 < 10:
            agent_hand_display += ' '
        print(agent_hand_display + ' ' + ' '.join(row))


if __name__=='__main__':
    toc = time.time()
    n_episodes = int(1e7)
    deck = initialize_deck()
    for ii in range(n_episodes):
        if len(deck) < 16:
            deck = initialize_deck()
        agent_state_action_pairs, reward, deck = play_episode(
            deck, Q, epsilon=0.01)
        Q, counter = update_Q_and_counter(
            Q, counter, agent_state_action_pairs, reward, gamma=0.9)
    print_stragegy_card(Q)
    print('{} episodes took {} seconds'.format(
        n_episodes, (time.time() - toc)))
    print(np.sum(counter, axis=-1))
