""" Blackjack learners """
import random
import math
from collections import Counter

import numpy as np

from blackjack.common import ACTIONS, AGENT_HANDS, AGENT_HAND_TO_INDEX

# pylint: disable=invalid-name
""" Q doesn't conform to snake_case naming style, but I felt it was
the most clear variable name for the action-value function """


class BaseLearner():
    """ Base learner class """

    def __init__(self, epsilon=0.1, window_size=1000):
        self.epsilon = epsilon
        self.Q = 0.1*np.random.rand(
            len(AGENT_HANDS), 10, len(ACTIONS))
        self.window_size = window_size
        self.reward_window = []
        self.windowed_training_rewards = []
        self.n_training_episodes = 0
        self.evaluations = []

    @staticmethod
    def cards_to_hand(cards):
        """ Convert a players list of cards into their 'hand', which
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

    @staticmethod
    def initialize_deck():
        """ Shuffle the deck """
        deck = 4*([2, 3, 4, 5, 6, 7, 8, 9, 'A'] + 4*[10])
        random.shuffle(deck)
        return deck

    @classmethod
    def deal(cls, deck):
        """ Deal cards """
        if len(deck) < 16:
            deck = cls.initialize_deck()
        print(deck)
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

    @classmethod
    def agent_state_to_index(cls, agent_state):
        """ convert agent state into state index """
        agent_hand, dealer_up_card = agent_state
        return (AGENT_HAND_TO_INDEX[agent_hand],
                cls.dealer_up_card_to_index(dealer_up_card))

    @classmethod
    def make_obvious_hits(cls, agent_cards, deck):
        """ It clearly makes no sense to stay if you have 11 or less """
        agent_hand = cls.cards_to_hand(agent_cards)
        while agent_hand[1] <= 11:
            agent_cards.append(deck.pop())
            agent_hand = cls.cards_to_hand(agent_cards)
        return agent_cards, agent_hand, deck

    def choose_epsilon_greedy_action(self, agent_state_index, epsilon=0.1):
        """ Choose an "epsilon-greedy" action. A fraction epsilon of
        the time, choose an action totally at random. Otherwise, choose
        an action greedily with respect to Q """
        if random.random() < epsilon:
            return random.randint(0, len(ACTIONS)-1)
        return np.argmax(self.Q[agent_state_index])

    def sample_agent_state_action(
            self, agent_cards, dealer_up_card, epsilon=0.1):
        """ Convert cards to agent state and sample an
        epsilon-greedy action """
        agent_hand = self.cards_to_hand(agent_cards)
        agent_state = (agent_hand, dealer_up_card)
        agent_state_index = self.agent_state_to_index(agent_state)
        action_index = self.choose_epsilon_greedy_action(
            agent_state_index, epsilon=epsilon)
        action = ACTIONS[action_index]
        return agent_state_index, action_index, action

    @staticmethod
    def evaluate_reward(agent_hand, dealer_hand):
        """ Evaluate the reward based on the agent and dealer hands """
        agent_sum, dealer_sum = agent_hand[1], dealer_hand[1]
        if agent_sum > 21:
            return -1
        if dealer_sum > 21:
            return 1
        if agent_sum == dealer_sum:
            return 0
        if agent_sum > dealer_sum:
            return 1
        return -1

    def print_stragegy_card(self):
        """ Print the strategy card according to the learned
        action-state value function, Q """
        card_actions = np.array(ACTIONS)[np.argmax(self.Q, axis=-1)]
        print('     ' + ' '.join(map(str, range(2, 11))) + 'A')
        for agent_hand, row in zip(
                reversed(AGENT_HANDS), reversed(card_actions)):
            print(' '.join(map(str, agent_hand)) + ' ' + ' '.join(row))

    def update_Q(self, agent_state_index, action_index):
        """ placeholder method for updating Q based on state, action,
        and observed reward """
        raise NotImplementedError(
            """Learner must implement function for updating state-action
            value function Q""")

    def play_agent_hand(self, agent_cards, dealer_up_card, deck, epsilon=0.1):
        """ Play out the agent's hand, updating Q at each action """
        agent_cards, agent_hand, deck = self.make_obvious_hits(
            agent_cards, deck)
        action = 'NONE'
        while (action != 'S') and (agent_hand[1] <= 21):
            agent_state_index, action_index, action = \
                self.sample_agent_state_action(
                    agent_cards, dealer_up_card, epsilon=epsilon)
            self.update_Q(agent_state_index, action_index)
            if action == 'H':
                agent_cards.append(deck.pop())
                agent_hand = self.cards_to_hand(agent_cards)
        return agent_hand, deck


    def play_episode(self, deck, epsilon=0.1):
        """ play out episode (ie, play out hand and see if the agent
        wins, loses, or pushes) """
        agent_cards, dealer_up_card, dealer_down_card, deck = self.deal(deck)
        dealer_cards = [dealer_up_card, dealer_down_card]
        agent_hand, deck = self.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=epsilon)
        dealer_hand, deck = self.play_dealer_hand(dealer_cards, deck)
        reward = self.evaluate_reward(agent_hand, dealer_hand)
        return reward, deck

    def evaluate_strategy(self, n_episodes=1000):
        """ Take greedy actions for a number of episodes, observe the
        rewards, and from this estimate the mean reward of this strategy."""
        deck = self.initialize_deck()
        reward_counts = Counter()
        for _ in range(n_episodes):
            _, reward, deck = self.play_episode(deck, epsilon=0)
            reward_counts[reward] += 1
        return self.mean_and_confidence_interval_from_counts(reward_counts)

    @classmethod
    def mean_and_confidence_interval_from_counts(cls, reward_counts):
        """ given counts of how many times each reward was received,
        estimate the mean reward and a 95% confidence interval for
        that estimated mean reward """
        n_episodes = sum(reward_counts)
        reward_values = list(reward_counts.keys()) # reward amounts
        # estimated multinomial probabilities of getting each reward value
        reward_phats = [count/n_episodes for count in reward_counts.values()]
        estimated_mean_reward = sum(
            reward*phat for reward, phat in zip(reward_values, reward_phats))
        estimated_variance = sum(
            (phat/n_episodes)*(reward - estimated_mean_reward)**2
            for reward, phat in zip(reward_values, reward_phats))
        confidence_interval = cls.variance_to_confidence_interval(
            estimated_variance)
        return estimated_mean_reward, confidence_interval

    @staticmethod
    def variance_to_confidence_interval(estimated_variance):
        """ given estimated variance, return 95% confidence interval +- """
        estimated_mean_std = math.sqrt(estimated_variance)
        confidence_interval = 1.96*estimated_mean_std
        return confidence_interval

    def train(self, n_episodes):
        """ Train the learner by plyaing n_episodes number of episodes """
        deck = self.initialize_deck()
        for _ in range(n_episodes):
            reward, deck = self.play_episode(deck, epsilon=self.epsilon)
            self.update_reward_history(reward)
            self.n_training_episodes += 1
        return self

    def update_reward_history(self, reward):
        """ Update the learners reward history with the latest reward.
        Here, we also keep track of the window-averaged reward. """
        self.reward_window.append(reward)
        if len(self.reward_window) >= self.window_size:
            self.windowed_training_rewards.append(np.mean(self.reward_window))
            self.reward_window = []

    def train_and_evaluate(self, n_episodes, n_evaluate_episodes=1000):
        """ Train, ane evaluate stragegy every time the number of
        training episodes doubles """
        episode_counter = 0
        while episode_counter < n_episodes:
            n_episodes_to_double_training_episodes = 2**math.ceil(
                np.log2(self.n_training_episodes + 0.1))
            n_episodes_to_train = min(
                n_episodes - episode_counter,
                n_episodes_to_double_training_episodes)
            self.train(n_episodes_to_train)
            self.evaluations.append(self.evaluate_strategy(
                n_episodes=n_evaluate_episodes))
