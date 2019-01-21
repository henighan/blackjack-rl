import main
import numpy as np
HIT_IND = main.ACTIONS.index('H')
STAY_IND = main.ACTIONS.index('S')


def test_initialize_deck_smoke():
    """ list each card, for each of the 4 suits, and shuffle """
    deck = 4*([2, 3, 4, 5, 6, 7, 8, 9, 'A'] + 4*[10])
    ret = main.initialize_deck()
    assert len(ret) == 52
    assert set(ret) == set(range(2, 11)).union(['A'])


def test_cards_to_hand_no_ace():
    agent_cards = [2, 10, 5]
    agent_hand = (' ', 17)
    ret = main.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_usable_ace():
    agent_cards = [2, 'A', 5]
    agent_hand = ('A', 18)
    ret = main.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_one_unusable_ace():
    agent_cards = [2, 10, 'A']
    agent_hand = (' ', 13)
    ret = main.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_usable():
    agent_cards = [2, 5, 'A', 'A']
    agent_hand = ('A', 19)
    ret = main.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_unusable():
    agent_cards = [2, 5, 'A', 'A', 3]
    agent_hand = (' ', 12)
    ret = main.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_dealer_up_card_to_index():
    dealer_up_card = 2
    index = 0
    ret = main.dealer_up_card_to_index(dealer_up_card)
    assert index == ret
    dealer_up_card = 10
    index = 8
    ret = main.dealer_up_card_to_index(dealer_up_card)
    assert index == ret


def test_agent_state_to_index_smoke():
    agent_hand = ('A', 13)
    dealer_up_card = 7
    agent_state = (agent_hand, dealer_up_card)
    assert main.agent_state_to_index(agent_state) == (
            main.AGENT_HAND_TO_INDEX[agent_hand],
            main.dealer_up_card_to_index(dealer_up_card))


def test_deal_smoke():
    deck = [1, 2, 3, 4]
    agent_cards = [4, 2]
    dealer_up_card = 3
    dealer_down_card = 1
    expected = (agent_cards, dealer_up_card, dealer_down_card, [])
    assert main.deal(deck) == expected


def test_play_dealer_hand_stay():
    dealer_cards = [10, 10]
    deck = [2, 3]
    dealer_hand = (' ', 20)
    assert main.play_dealer_hand(dealer_cards, deck) == (dealer_hand, deck)


def test_play_dealer_hand_hit():
    dealer_cards = [10, 5]
    deck = [2, 3]
    played_dealer_hand = (' ', 18)
    played_deck = [2]
    ret = main.play_dealer_hand(dealer_cards, deck)
    assert (played_dealer_hand, played_deck) == ret


def test_play_agent_hand_stay(mocker):
    agent_cards = [10, 10]
    agent_hand = (' ', 20)
    dealer_up_card = 10
    deck = [] # since we're staying, no cards should be drawn from the deck
    # initialize Q so the highest-value action is 'Stay' for all states
    Q = np.zeros([1, 1, len(main.ACTIONS)])
    Q[:, :, STAY_IND] = 1
    agent_state_index = (0, 0)
    agent_state_action_pairs = [(agent_state_index, STAY_IND)]
    with mocker.patch.object(
            main, 'agent_state_to_index', return_value=agent_state_index):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            main.monte_carlo_play_agent_hand(
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
    Q = np.zeros([2, 2, len(main.ACTIONS)])
    Q[first_agent_state][HIT_IND] = 1
    Q[second_agent_state][STAY_IND] = 1
    agent_state_action_pairs = [
        (first_agent_state, HIT_IND), (second_agent_state, STAY_IND)]
    deck = [2, 3]
    with mocker.patch.object(
            main, 'agent_state_to_index',
            side_effect=[first_agent_state, second_agent_state]):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            main.monte_carlo_play_agent_hand(
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
    Q = np.zeros([1, 1, len(main.ACTIONS)])
    Q[:, :, HIT_IND] = 1
    agent_state_action_pairs = [(agent_state, HIT_IND)]
    deck = [6, 6]
    with mocker.patch.object(
            main, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            main.monte_carlo_play_agent_hand(
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
    mocker.patch('main.np.random.randint', return_value=HIT_IND)
    with mocker.patch.object(
            main, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            main.monte_carlo_play_agent_hand(
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
    mocker.patch('main.np.random.randint', return_value=STAY_IND)
    with mocker.patch.object(
            main, 'agent_state_to_index', return_value=agent_state):
        ret_agent_hand, ret_agent_state_action_pairs, ret_deck = \
            main.monte_carlo_play_agent_hand(
                agent_cards, dealer_up_card, Q, deck, epsilon=1)
    assert ret_agent_hand == (' ', 20)
    assert ret_agent_state_action_pairs, agent_state_action_pairs
    assert ret_deck == []


def test_evaluate_reward_agent_bust():
    agent_hand = (' ', 22)
    dealer_hand = (' ', 20)
    assert main.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_dealer_bust():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 22)
    assert main.evaluate_reward(agent_hand, dealer_hand) == 1


def test_evaluate_reward_dealer_wins():
    agent_hand = (' ', 17)
    dealer_hand = (' ', 20)
    assert main.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_agent_wins():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 17)
    assert main.evaluate_reward(agent_hand, dealer_hand) == 1

def test_evaluate_reward_push():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 20)
    assert main.evaluate_reward(agent_hand, dealer_hand) == 0


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
    ret_Q, ret_counter = main.monte_carlo_update_Q_and_counter(
        Q, counter, agent_state_action_pairs, 1, gamma=gamma)
    np.testing.assert_equal(ret_counter, counter)
    np.testing.assert_equal(ret_Q, Q)
