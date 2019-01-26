import common
import numpy as np


HIT_IND = common.ACTIONS.index('H')
STAY_IND = common.ACTIONS.index('S')


def test_initialize_deck_smoke():
    """ list each card, for each of the 4 suits, and shuffle """
    deck = 4*([2, 3, 4, 5, 6, 7, 8, 9, 'A'] + 4*[10])
    ret = common.initialize_deck()
    assert len(ret) == 52
    assert set(ret) == set(range(2, 11)).union(['A'])


def test_cards_to_hand_no_ace():
    agent_cards = [2, 10, 5]
    agent_hand = (' ', 17)
    ret = common.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_usable_ace():
    agent_cards = [2, 'A', 5]
    agent_hand = ('A', 18)
    ret = common.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_one_unusable_ace():
    agent_cards = [2, 10, 'A']
    agent_hand = (' ', 13)
    ret = common.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_usable():
    agent_cards = [2, 5, 'A', 'A']
    agent_hand = ('A', 19)
    ret = common.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_cards_to_hand_two_aces_unusable():
    agent_cards = [2, 5, 'A', 'A', 3]
    agent_hand = (' ', 12)
    ret = common.cards_to_hand(agent_cards)
    assert ret == agent_hand


def test_dealer_up_card_to_index():
    dealer_up_card = 2
    index = 0
    ret = common.dealer_up_card_to_index(dealer_up_card)
    assert index == ret
    dealer_up_card = 10
    index = 8
    ret = common.dealer_up_card_to_index(dealer_up_card)
    assert index == ret


def test_agent_state_to_index_smoke():
    agent_hand = ('A', 13)
    dealer_up_card = 7
    agent_state = (agent_hand, dealer_up_card)
    assert common.agent_state_to_index(agent_state) == (
            common.AGENT_HAND_TO_INDEX[agent_hand],
            common.dealer_up_card_to_index(dealer_up_card))


def test_deal_smoke():
    deck = [1, 2, 3, 4]
    agent_cards = [4, 2]
    dealer_up_card = 3
    dealer_down_card = 1
    expected = (agent_cards, dealer_up_card, dealer_down_card, [])
    assert common.deal(deck) == expected


def test_play_dealer_hand_stay():
    dealer_cards = [10, 10]
    deck = [2, 3]
    dealer_hand = (' ', 20)
    assert common.play_dealer_hand(dealer_cards, deck) == (dealer_hand, deck)


def test_play_dealer_hand_hit():
    dealer_cards = [10, 5]
    deck = [2, 3]
    played_dealer_hand = (' ', 18)
    played_deck = [2]
    ret = common.play_dealer_hand(dealer_cards, deck)
    assert (played_dealer_hand, played_deck) == ret


def test_evaluate_reward_agent_bust():
    agent_hand = (' ', 22)
    dealer_hand = (' ', 20)
    assert common.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_dealer_bust():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 22)
    assert common.evaluate_reward(agent_hand, dealer_hand) == 1


def test_evaluate_reward_dealer_wins():
    agent_hand = (' ', 17)
    dealer_hand = (' ', 20)
    assert common.evaluate_reward(agent_hand, dealer_hand) == -1


def test_evaluate_reward_agent_wins():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 17)
    assert common.evaluate_reward(agent_hand, dealer_hand) == 1

def test_evaluate_reward_push():
    agent_hand = (' ', 20)
    dealer_hand = (' ', 20)
    assert common.evaluate_reward(agent_hand, dealer_hand) == 0


def test_sample_agent_state_action_smoke(mocker):
    agent_cards = [10, 2]
    dealer_up_card = 10
    with mocker.patch.object(common, 'choose_epsilon_greedy_action',
                             return_value=HIT_IND):
        ret_state_index, ret_action_index, ret_action = \
            common.sample_agent_state_action(agent_cards, dealer_up_card, None)
    assert ret_action_index == HIT_IND
    assert ret_action == 'H'
    agent_hand = common.cards_to_hand(agent_cards)
    agent_state = (agent_hand, dealer_up_card)
    agent_state_index = common.agent_state_to_index(agent_state)
    assert ret_state_index == agent_state_index


def test_choose_epsilon_greedy_action_hit():
    agent_state_index = (0, 0)
    Q = np.zeros([1, 1, 2])
    Q[:,:,HIT_IND] = 1
    action_index = common.choose_epsilon_greedy_action(
        Q, agent_state_index, epsilon=0)
    assert action_index == HIT_IND


def test_choose_epsilon_greedy_action_stay():
    agent_state_index = (0, 0)
    Q = np.zeros([1, 1, 2])
    Q[:, :, STAY_IND] = 1
    action_index = common.choose_epsilon_greedy_action(
        Q, agent_state_index, epsilon=0)
    assert action_index == STAY_IND


def test_choose_epsilon_greedy_action_random(mocker):
    agent_state_index = None
    Q = None
    mocker.patch('common.np.random.randint', return_value=STAY_IND)
    action_index = common.choose_epsilon_greedy_action(
        Q, agent_state_index, epsilon=1)
    assert action_index == STAY_IND


def test_make_obvious_hits_no_hits():
    agent_cards = [10, 2]
    deck = [] # no cards should be taken from the deck
    agent_hand = common.cards_to_hand(agent_cards)
    ret_cards, ret_hand, deck = common.make_obvious_hits(agent_cards, deck)
    assert ret_cards == [10, 2]
    assert ret_hand == agent_hand
    assert deck == []
