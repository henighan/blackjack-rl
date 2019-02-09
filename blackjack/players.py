""" The players in this game of blackjack (the dealer, and the agent) """
from random import shuffle
from collections import Counter

import numpy as np

from blackjack.common import (ACE_HANDS, ACTIONS, AGENT_HANDS,
                              AGENT_HAND_TO_INDEX, NO_ACE_HANDS)

class BasePlayer(object):
    """ Methods used both by the dealer and the agent """

    def __init__(self):
        pass

    @staticmethod
    def cards_to_hand(cards):
        """ Convert a players list of cards into their hand, which
        is a tuple specifying the current sum of their cards, and
        whether or not they have a usable Ace """
        n_aces = cards.count('A')
        non_ace_sum = sum(card for card in cards if card != 'A')
        if n_aces == 0:
            return (' ', non_ace_sum)
        usable_ace_sum = (n_aces - 1)*1 + 11 + non_ace_sum
        if usable_ace_sum <= 21:
            return ('A', usable_ace_sum)
        return (' ', n_aces*1 + non_ace_sum)


class Dealer(BasePlayer):
    """ Methods pertaining to the blackjack dealer """

    def __init__(self):
        # pylint: disable=missing-super-argument
        super().__init__(self)

    @staticmethod
    def initialize_deck():
        """ Shuffle the deck """
        deck = 4*([2, 3, 4, 5, 6, 7, 8, 9, 'A'] + 4*[10])
        shuffle(deck)
        return deck

    @staticmethod
    def deal(deck):
        """ Deal cards """
        agent_cards = [deck.pop()]
        dealer_up_card = deck.pop()
        agent_cards.append(deck.pop())
        dealer_down_card = deck.pop()
        return agent_cards, dealer_up_card, dealer_down_card, deck

    @staticmethod
    def dealer_up_card_to_index(dealer_up_card):
        """ Convert the dealer_up_card into the state index in Q """
        if dealer_up_card == 'A':
            return 9
        return dealer_up_card - 2

    @classmethod
    def play_dealer_hand(cls, dealer_cards, deck):
        """ Play out the dealer's hand """
        dealer_hand = cls.cards_to_hand(dealer_cards)
        while dealer_hand[1] < 17:
            dealer_cards.append(deck.pop())
            dealer_hand = cls.cards_to_hand(dealer_cards)
        return dealer_hand, deck


class Agent(BasePlayer):
    """ Methods used by the agent, which is playing blackjack """

    def __init__(self):
        # pylint: disable=missing-super-argument
        super().__init__(self)

    @classmethod
    def make_obvious_hits(cls, agent_cards, deck):
        """ It clearly makes no sense to stay if you have 11 or less """
        agent_hand = cls.cards_to_hand(agent_cards)
        while agent_hand[1] <= 11:
            agent_cards.append(deck.pop())
            agent_hand = cls.cards_to_hand(agent_cards)
        return agent_cards, agent_hand, deck

    def choose_epsilon_greedy_action(self, agent_state_index, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(ACTIONS))
        return np.argmax(self.Q[agent_state_index])
