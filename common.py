from random import shuffle
import numpy as np


ACTIONS = ['S', 'H'] # S: stay, H: hit
NO_ACE_HANDS = [(' ', card_sum) for card_sum in range(12, 22)]
ACE_HANDS = [('A', card_sum) for card_sum in range(12, 22)]
AGENT_HANDS = NO_ACE_HANDS + ACE_HANDS
AGENT_HAND_TO_INDEX = {hand: index for index, hand in enumerate(AGENT_HANDS)}


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


def sample_agent_state_action(agent_cards, dealer_up_card, Q, epsilon=0.1):
    agent_hand = cards_to_hand(agent_cards)
    agent_state = (agent_hand, dealer_up_card)
    agent_state_index = agent_state_to_index(agent_state)
    action_index = choose_epsilon_greedy_action(
        Q, agent_state_index, epsilon=epsilon)
    action = ACTIONS[action_index]
    return agent_state_index, action_index, action


def sarsa_play_episode(agent_cards, dealer_up_card, dealer_down_card, Q, deck,
                        alpha=0.05, gamma=0.9, epsilon=0.1):
    agent_cards, agent_hand = make_obvious_hits(agent_cards)
    agent_state_index, action_index, action = sample_agent_state_action(
        agent_cards, dealer_up_card, Q, epsilon=epsilon)
    while (action != 'S') and (agent_hand[1] <= 21):
        if action == 'H':
            agent_cards.append(deck.pop())
            agent_hand = cards_to_hand(agent_cards)
        next_agent_state_index, next_action_index, next_action = \
            sample_agent_state_action(agent_cards, dealer_up_card,
                                      Q, epsilon=epsilon)
        Q = sarsa_update_Q(
            Q, agent_state_index, action_index,
            next_agent_state_index, next_action_index, alpha=alpha,
            gamma=gamma)
        agent_state_index = next_agent_state_index
        action_index, action = next_action_index, next_action
    dealer_cards = [dealer_up_card, dealer_down_card]
    dealer_hand, deck = play_dealer_hand(dealer_cards, deck)
    reward = evaluate_reward(agent_hand, dealer_hand)
    Q[agent_state_index][action_index] += alpha*(
        reward - Q[agent_state_index][action_index])
    return Q, reward, deck


def make_obvious_hits(agent_cards, deck):
    """ It clearly makes no sense to stay if you have 11 or less """
    agent_hand = cards_to_hand(agent_cards)
    while agent_hand[1] <= 11:
        agent_cards.append(deck.pop())
        agent_hand = cards_to_hand(agent_cards)
    return agent_cards, agent_hand, deck


def choose_epsilon_greedy_action(Q, agent_state_index, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(ACTIONS))
    return np.argmax(Q[agent_state_index])


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


def print_stragegy_card(Q):
    card_actions = np.array(ACTIONS)[np.argmax(Q, axis=-1)]
    print('     ' + ' '.join(map(str, range(2, 11))) + 'A')
    for agent_hand, row in zip(reversed(AGENT_HANDS), reversed(card_actions)):
        print(' '.join(map(str, agent_hand)) + ' ' + ' '.join(row))
