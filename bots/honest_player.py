import sys
import json
import scipy.stats as sps
import numpy as np
import bots.commons as commons

from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


NB_SIMULATION = 300

class HonestPlayer(BasePokerPlayer):

    def __init__(self, nb_simulation=NB_SIMULATION):
        self.nb_simulation = nb_simulation

    def declare_action(self, valid_actions, hole_card, round_state):
        # self.nb_player = len([[player for player in players if player.is_active()]])
        self.nb_active = len([player for player in round_state['seats'] if player['state'] != 'folded'])
        community_card = round_state['community_card']
        win_rate =  commons.estimate_hole_card_win_rate(
                nb_simulation=self.nb_simulation,
                nb_player=self.nb_active,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
        if win_rate >= 1.0 / self.nb_active:
            action = valid_actions[1]  # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info


        # print('all players: {}, active players: {}'.format(self.nb_player,
        # len([player for player in players if player.is_active()])), file=sys.stderr)


        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass



if __name__ == '__main__':

    player = HonestPlayer()

    while True:
        line = sys.stdin.readline().rstrip()
        if not line:
           break
        event_type, data = line.split('\t', 1)
        data = json.loads(data)

        if event_type == 'declare_action':
            action, amount = player.declare_action(data['valid_actions'], data['hole_card'], data['round_state'])
            sys.stdout.write('{}\t{}\n'.format(action, amount))
            sys.stdout.flush()
        elif event_type == 'game_start':
            player.receive_game_start_message(data)
        elif event_type == 'round_start':
            player.receive_round_start_message(data['round_count'], data['hole_card'], data['seats'])
        elif event_type == 'street_start':
            player.receive_street_start_message(data['street'], data['round_state'])
        elif event_type == 'game_update':
            player.receive_game_update_message(data['new_action'], data['round_state'])
        elif event_type == 'round_result':
            player.receive_round_result_message(data['winners'], data['hand_info'], data['round_state'])
        else:
            raise RuntimeError('Bad event type "{}"'.format(event_type))
