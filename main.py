import time
from random import shuffle
import numpy as np


ACTIONS = ['S', 'H'] # S: stay, H: hit
NO_ACE_HANDS = [(' ', card_sum) for card_sum in range(12, 22)]
ACE_HANDS = [('A', card_sum) for card_sum in range(12, 22)]
AGENT_HANDS = NO_ACE_HANDS + ACE_HANDS
AGENT_HAND_TO_INDEX = {hand: index for index, hand in enumerate(AGENT_HANDS)}
Q = 0.1*np.random.rand(len(AGENT_HANDS), 10, len(ACTIONS))
counter = np.zeros_like(Q)


def initialize_deck():
    deck = 4*([2, 3, 4, 5, 6, 7, 8, 9, 'A'] + 4*[10])
    shuffle(deck)
    return deck


def cards_to_hand(cards):
    n_aces = cards.count('A')
    non_ace_sum = sum(card for card in cards if card != 'A')
    if n_aces == 0:
        return (' ', non_ace_sum)
    usable_ace_sum = (n_aces - 1)*1 + 11 + non_ace_sum
    if usable_ace_sum <= 21:
        return ('A', usable_ace_sum)
    return (' ', n_aces*1 + non_ace_sum)


def dealer_up_card_to_index(dealer_up_card):
    if dealer_up_card == 'A':
        return 9
    return dealer_up_card - 2

def agent_state_to_index(agent_state):
    agent_hand, dealer_up_card = agent_state
    return AGENT_HAND_TO_INDEX[agent_hand], dealer_up_card_to_index(dealer_up_card)


def deal(deck):
    agent_cards = [deck.pop()]
    dealer_up_card = deck.pop()
    agent_cards.append(deck.pop())
    dealer_down_card = deck.pop()
    return agent_cards, dealer_up_card, dealer_down_card, deck


def play_agent_hand(agent_cards, dealer_up_card, Q, deck, epsilon=0.1):
    agent_hand = cards_to_hand(agent_cards)
    while agent_hand[1] <= 11:
        """ It clearly makes no sense to stay if you have 11 or less """
        agent_cards.append(deck.pop())
        agent_hand = cards_to_hand(agent_cards)
    agent_state_action_pairs = []
    action = 'H'
    while (action != 'S') and (agent_hand[1] <= 21):
        agent_state = (agent_hand, dealer_up_card)
        agent_state_index = agent_state_to_index(agent_state)
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, Q.shape[-1])
        else:
            action_index = np.argmax(Q[agent_state_index])
        agent_state_action_pairs.append((agent_state_index, action_index))
        action = ACTIONS[action_index]
        if action == 'H':
            agent_cards.append(deck.pop())
            agent_hand = cards_to_hand(agent_cards)
    return agent_hand, agent_state_action_pairs, deck


def play_dealer_hand(dealer_cards, deck):
    dealer_hand = cards_to_hand(dealer_cards)
    while dealer_hand[1] < 17:
        dealer_cards.append(deck.pop())
        dealer_hand = cards_to_hand(dealer_cards)
    return dealer_hand, deck


def evaluate_reward(agent_hand, dealer_hand):
    agent_sum, dealer_sum = agent_hand[1], dealer_hand[1]
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
    agent_cards, dealer_up_card, dealer_down_card, deck = deal(deck)
    dealer_cards = [dealer_up_card, dealer_down_card]
    agent_hand, agent_state_action_pairs, deck = play_agent_hand(
        agent_cards, dealer_up_card, Q, deck, epsilon=epsilon)
    dealer_hand, deck = play_dealer_hand(dealer_cards, deck)
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
    print('     ' + ' '.join(map(str, range(2, 11))) + 'A')
    for agent_hand, row in zip(reversed(AGENT_HANDS), reversed(card_actions)):
        print(' '.join(map(str, agent_hand)) + ' ' + ' '.join(row))


if __name__=='__main__':
    toc = time.time()
    n_episodes = int(1e5)
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
    # print(np.sum(counter, axis=-1))
