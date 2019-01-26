import numpy as np
import common
import monte_carlo
HIT_IND = common.ACTIONS.index('H')
STAY_IND = common.ACTIONS.index('S')


def test_play_agent_hand_stay(mocker):
    agent_cards = [10, 10]
    agent_hand = (' ', 20)
    dealer_up_card = 10
    deck = [] # since we're staying, no cards should be drawn from the deck
    # initialize Q so the highest-value action is 'Stay' for all states
    Q = np.zeros([1, 1, len(common.ACTIONS)])
    Q[:, :, STAY_IND] = 1
    agent_state_index = (0, 0)
    agent_state_action_pairs = [(agent_state_index, STAY_IND)]
    with mocker.patch.object(
            common, 'agent_state_to_index', return_value=agent_state_index):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            monte_carlo.play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=0)
    assert ret_agent_hand == agent_hand
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == deck


def test_play_agent_hand_hit_stay(mocker):
    agent_cards = [10, 5]
    dealer_up_card = 10
    first_agent_state = (0, 0)
    second_agent_state = (1, 1)
    # initialize Q so the highest-value action is 'Hit' for first state,
    # 'Stay' for second
    Q = np.zeros([2, 2, len(common.ACTIONS)])
    Q[first_agent_state][HIT_IND] = 1
    Q[second_agent_state][STAY_IND] = 1
    agent_state_action_pairs = [
        (first_agent_state, HIT_IND), (second_agent_state, STAY_IND)]
    deck = [2, 3]
    with mocker.patch.object(
            common, 'agent_state_to_index',
            side_effect=[first_agent_state, second_agent_state]):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            monte_carlo.play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=0)
    assert ret_agent_hand == (' ', 18)
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == [2]



def test_play_agent_hand_hit_bust(mocker):
    agent_cards = [10, 6]
    dealer_up_card = 10
    agent_state = (0, 0)
    # initialize Q so the highest-value action is 'Hit' for first state,
    # 'Stay' for second
    Q = np.zeros([1, 1, len(common.ACTIONS)])
    Q[:, :, HIT_IND] = 1
    agent_state_action_pairs = [(agent_state, HIT_IND)]
    deck = [6, 6]
    with mocker.patch.object(
            common, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            monte_carlo.play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=0)
    assert ret_agent_hand == (' ', 22)
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == [6]


def test_play_agent_hand_random_hit_bust(mocker):
    agent_cards = [10, 6]
    dealer_up_card = 10
    agent_state = (0, 0)
    Q = np.array([0])
    agent_state_action_pairs = [(agent_state, HIT_IND)]
    deck = [6, 6]
    mocker.patch('common.np.random.randint', return_value=HIT_IND)
    with mocker.patch.object(
            common, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            monte_carlo.play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=1)
    assert ret_agent_hand == (' ', 22)
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == [6]

def test_play_agent_hand_random_stay(mocker):
    agent_cards = [10, 10]
    dealer_up_card = 10
    agent_state = (0, 0)
    Q = np.array([0])
    agent_state_action_pairs = [(agent_state, STAY_IND)]
    deck = []
    mocker.patch('common.np.random.randint', return_value=STAY_IND)
    with mocker.patch.object(
            common, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            monte_carlo.play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=1)
    assert ret_agent_hand == (' ', 20)
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == []


def test_update_Q_smoke():
    Q = np.zeros([2, 2, 2])
    counter = np.zeros([2, 2, 2])
    counter = np.zeros([2, 2, 2])
    agent_state_action_pairs = [((0, 0), 0), ((1, 1), 1)]
    reward = 1
    gamma = 0.9
    updated_counter = np.zeros([2, 2, 2])
    updated_counter[0, 0, 0] = 1
    updated_counter[1, 1, 1] = 1
    updated_Q = np.zeros([2, 2, 2])
    updated_Q[0, 0, 0] = gamma*reward
    updated_Q[1, 1, 1] = reward
    ret_Q, ret_counter = monte_carlo.update_Q_and_counter(
        Q, counter, agent_state_action_pairs, 1, gamma=gamma)
    np.testing.assert_equal(ret_counter, counter)
    np.testing.assert_equal(ret_Q, Q)
