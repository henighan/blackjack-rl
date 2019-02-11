""" Tests for Base Learner """
import numpy as np

import common


HIT_IND = common.ACTIONS.index('H')
STAY_IND = common.ACTIONS.index('S')


def test_initialize_deck_smoke(base_learner):
    """ list each card, for each of the 4 suits, and shuffle """
    ret = base_learner.initialize_deck()
    assert len(ret) == 52
    assert set(ret) == set(range(2, 11)).union(['A'])


def test_cards_to_hand_no_ace(base_learner):
    """ Test converting card list to 'hand' when no ace is present """
    agent_cards = [2, 10, 5]
    agent_hand = (' ', 17)
    ret = base_learner.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_usable_ace(base_learner):
    """ Test converting card list to 'hand' when usable ace is present """
    agent_cards = [2, 'A', 5]
    agent_hand = ('A', 18)
    ret = base_learner.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_one_unusable_ace(base_learner):
    """ Test converting card list to 'hand' when UNusable ace is present """
    agent_cards = [2, 10, 'A']
    agent_hand = (' ', 13)
    ret = base_learner.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_usable(base_learner):
    """ Test converting card list to 'hand' when there's two aces, and
    one of them is usable """
    agent_cards = [2, 5, 'A', 'A']
    agent_hand = ('A', 19)
    ret = base_learner.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_unusable(base_learner):
    """ Test converting card list to 'hand' when there's two aces, and
    neither of them is usable """
    agent_cards = [2, 5, 'A', 'A', 3]
    agent_hand = (' ', 12)
    ret = base_learner.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_dealer_up_card_to_index(base_learner):
    """ Test converting dealer up card to index, for indexing agent
    states in Q """
    dealer_up_card = 2
    index = 0
    ret = base_learner.dealer_up_card_to_index(dealer_up_card)
    assert index == ret
    dealer_up_card = 10
    index = 8
    ret = base_learner.dealer_up_card_to_index(dealer_up_card)
    assert index == ret


def test_agent_state_to_index_smoke(base_learner):
    """" Test converting agent state (a tuple of agent_hand and
    dealer_up_card) to index in Q """
    agent_hand = ('A', 13)
    dealer_up_card = 7
    agent_state = (agent_hand, dealer_up_card)
    assert base_learner.agent_state_to_index(agent_state) == (
        common.AGENT_HAND_TO_INDEX[agent_hand],
        base_learner.dealer_up_card_to_index(dealer_up_card))


def test_deal_smoke(mocker, base_learner):
    """ Dealing smoke test """
    deck = [1, 2, 3, 4]
    agent_cards = [4, 2]
    dealer_up_card = 3
    dealer_down_card = 1
    mocker.patch(
        'blackjack.learners.BaseLearner.initialize_deck', return_value=deck)
    expected = (agent_cards, dealer_up_card, dealer_down_card, [])
    assert base_learner.deal(deck) == expected


def test_play_dealer_hand_stay(base_learner):
    """ test play_dealer_hand when the action is STAY """
    dealer_cards = [10, 10]
    deck = [2, 3]
    dealer_hand = (' ', 20)
    assert base_learner.play_dealer_hand(dealer_cards, deck) == (dealer_hand, deck)


def test_play_dealer_hand_hit(base_learner):
    """ test play_dealer_hand when the action is HIT """
    dealer_cards = [10, 5]
    deck = [2, 3]
    played_dealer_hand = (' ', 18)
    played_deck = [2]
    ret = base_learner.play_dealer_hand(dealer_cards, deck)
    assert (played_dealer_hand, played_deck) == ret


def test_evaluate_reward_agent_bust(base_learner):
    """ Test evaluating reward when the agent busted """
    agent_hand = (' ', 22)
    dealer_hand = (' ', 20)
    assert base_learner.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_dealer_bust(base_learner):
    """ Test evaluating reward when the dealer busted """
    agent_hand = (' ', 20)
    dealer_hand = (' ', 22)
    assert base_learner.evaluate_reward(agent_hand, dealer_hand) == 1


def test_evaluate_reward_dealer_wins(base_learner):
    """ Test evaluating reward when the dealer won """
    agent_hand = (' ', 17)
    dealer_hand = (' ', 20)
    assert base_learner.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_agent_wins(base_learner):
    """ Test evaluating reward when the agent won """
    agent_hand = (' ', 20)
    dealer_hand = (' ', 17)
    assert base_learner.evaluate_reward(agent_hand, dealer_hand) == 1

def test_evaluate_reward_push(base_learner):
    """ Test evaluating reward when its a push (tie) """
    agent_hand = (' ', 20)
    dealer_hand = (' ', 20)
    assert base_learner.evaluate_reward(agent_hand, dealer_hand) == 0


def test_sample_agent_state_action_smoke(mocker, base_learner):
    """ Smoke test for sampling agent state and action """
    agent_cards = [10, 2]
    dealer_up_card = 10
    with mocker.patch.object(base_learner, 'choose_epsilon_greedy_action',
                             return_value=HIT_IND):
        ret_state_index, ret_action_index, ret_action = \
            base_learner.sample_agent_state_action(
                agent_cards, dealer_up_card, None)
    assert ret_action_index == HIT_IND
    assert ret_action == 'H'
    agent_hand = base_learner.cards_to_hand(agent_cards)
    agent_state = (agent_hand, dealer_up_card)
    agent_state_index = base_learner.agent_state_to_index(agent_state)
    assert ret_state_index == agent_state_index


def test_choose_epsilon_greedy_action_hit(base_learner):
    """ test choose_epsilon_greedy_action select HIT greedily """
    agent_state_index = (0, 0)
    base_learner.Q = np.zeros([1, 1, 2])
    base_learner.Q[:, :, HIT_IND] = 1
    action_index = base_learner.choose_epsilon_greedy_action(
        agent_state_index, epsilon=0)
    assert action_index == HIT_IND


def test_choose_epsilon_greedy_action_stay(base_learner):
    """ test choose_epsilon_greedy_action select Stay greedily """
    agent_state_index = (0, 0)
    base_learner.Q = np.zeros([1, 1, 2])
    base_learner.Q[:, :, STAY_IND] = 1
    action_index = base_learner.choose_epsilon_greedy_action(
        agent_state_index, epsilon=0)
    assert action_index == STAY_IND


def test_choose_epsilon_greedy_action_random(mocker, base_learner):
    """ test choosing random action """
    agent_state_index = None
    base_learner.Q = None
    mocker.patch('blackjack.learners.random.randint', return_value=STAY_IND)
    action_index = base_learner.choose_epsilon_greedy_action(
        agent_state_index, epsilon=1)
    assert action_index == STAY_IND


def test_make_obvious_hits_no_hits(base_learner):
    """ Test agent making obvious hits when it shouldn't hit """
    agent_cards = [10, 2]
    deck = [] # no cards should be taken from the deck
    agent_hand = base_learner.cards_to_hand(agent_cards)
    ret_cards, ret_hand, deck = base_learner.make_obvious_hits(agent_cards, deck)
    assert ret_cards == [10, 2]
    assert ret_hand == agent_hand
    assert deck == []


def test_make_obvious_hits_one_hit(base_learner):
    """ Test agent making obvious hits when it should hit one time """
    agent_cards = [5, 2]
    deck = [5] # no cards should be taken from the deck
    agent_hand = base_learner.cards_to_hand(agent_cards + deck)
    ret_cards, ret_hand, deck = base_learner.make_obvious_hits(agent_cards, deck)
    assert ret_cards == [5, 2, 5]
    assert ret_hand == agent_hand
    assert deck == []


def test_make_obvious_hits_two_hits(base_learner):
    """ Test agent making obvious hits when it should hit twice """
    agent_cards = [5, 2]
    deck = [2, 3] # no cards should be taken from the deck
    # cards the agent should have after hits
    final_cards = [5, 2, 3, 2]
    agent_hand = base_learner.cards_to_hand(final_cards)
    ret_cards, ret_hand, deck = base_learner.make_obvious_hits(
        agent_cards, deck)
    assert ret_cards == final_cards
    assert ret_hand == agent_hand
    assert deck == []


def test_play_agent_hand_stay(mocker, base_learner):
    """ Test playing the agents hand when the greedy action is to stay """
    agent_cards = [10, 10]
    agent_hand = (' ', 20)
    dealer_up_card = 10
    deck = [] # since we're staying, no cards should be drawn from the deck
    # initialize Q so the highest-value action is 'Stay' for all states
    base_learner.Q = np.zeros([1, 1, len(common.ACTIONS)])
    base_learner.Q[:, :, STAY_IND] = 1
    agent_state_index = (0, 0)
    mocker.patch('blackjack.learners.BaseLearner.update_Q')
    with mocker.patch.object(
            base_learner, 'agent_state_to_index',
            return_value=agent_state_index):
        ret_agent_hand, ret_deck = base_learner.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=0)
    assert ret_agent_hand == agent_hand
    assert ret_deck == deck


def test_play_agent_hand_hit_stay(mocker, base_learner):
    """ Test playing the agents hand when the greedy actions are to hit,
    then stay. """
    agent_cards = [10, 5]
    dealer_up_card = 10
    first_agent_state = (0, 0)
    second_agent_state = (1, 1)
    # initialize Q so the highest-value action is 'Hit' for first state,
    # 'Stay' for second
    base_learner.Q = np.zeros([2, 2, len(common.ACTIONS)])
    base_learner.Q[first_agent_state][HIT_IND] = 1
    base_learner.Q[second_agent_state][STAY_IND] = 1
    deck = [2, 3]
    mocker.patch('blackjack.learners.BaseLearner.update_Q')
    with mocker.patch.object(
            base_learner, 'agent_state_to_index',
            side_effect=[first_agent_state, second_agent_state]):
        ret_agent_hand, ret_deck = base_learner.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=0)
    assert ret_agent_hand == (' ', 18)
    assert ret_deck == [2]


def test_play_agent_hand_hit_bust(mocker, base_learner):
    """ test play agent hand when the greedy action is to hit
    which results in a bust """
    agent_cards = [10, 6]
    dealer_up_card = 10
    agent_state = (0, 0)
    # initialize Q so the highest-value action is 'Hit' for first state,
    # 'Stay' for second
    base_learner.Q = np.zeros([1, 1, len(common.ACTIONS)])
    base_learner.Q[:, :, HIT_IND] = 1
    deck = [6, 6]
    mocker.patch('blackjack.learners.BaseLearner.update_Q')
    with mocker.patch.object(
            base_learner, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_deck = base_learner.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=0)
    assert ret_agent_hand == (' ', 22)
    assert ret_deck == [6]


def test_play_agent_hand_random_hit_bust(mocker, base_learner):
    """ test play agent hand when randomly chosen action is hit
    that leads to a bust """
    agent_cards = [10, 6]
    dealer_up_card = 10
    agent_state = (0, 0)
    base_learner.Q = np.array([0])
    deck = [6, 6]
    mocker.patch('blackjack.learners.BaseLearner.update_Q')
    mocker.patch('blackjack.learners.random.randint', return_value=HIT_IND)
    with mocker.patch.object(
            base_learner, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_deck = base_learner.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=1)
    assert ret_agent_hand == (' ', 22)
    assert ret_deck == [6]


def test_play_agent_hand_random_stay(mocker, base_learner):
    """ test play agent hand when randomly chosen action is stay """
    agent_cards = [10, 10]
    dealer_up_card = 10
    agent_state = (0, 0)
    base_learner.Q = np.array([0])
    deck = []
    mocker.patch('blackjack.learners.BaseLearner.update_Q')
    mocker.patch('blackjack.learners.random.randint', return_value=STAY_IND)
    with mocker.patch.object(
            base_learner, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_deck = base_learner.play_agent_hand(
            agent_cards, dealer_up_card, deck, epsilon=1)
    assert ret_agent_hand == (' ', 20)
    assert ret_deck == []
