""" Common/Global variables """
ACTIONS = ['S', 'H'] # S: stay, H: hit
NO_ACE_HANDS = [(' ', card_sum) for card_sum in range(12, 22)]
ACE_HANDS = [('A', card_sum) for card_sum in range(12, 22)]
AGENT_HANDS = NO_ACE_HANDS + ACE_HANDS
AGENT_HAND_TO_INDEX = {hand: index for index, hand in enumerate(AGENT_HANDS)}
