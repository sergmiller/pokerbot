import sys
import json
import pprint
import numpy as np
import scipy.stats as sps

try:
    import commons
except:
    from . import commons

import pprint

from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

from copy import deepcopy

from keras.optimizers import SGD
from keras.losses import mse

from collections import deque


from keras.models import load_model



def load(model_file):
    model = load_model(model_file)
    sgd = SGD(lr=1e-2,
              momentum=0.95,
              decay=1,
              nesterov=True,
             )
    model.compile(loss=mse,
                  optimizer=sgd,
                 )
    return model


class MergePlayer(BasePokerPlayer):
    def __init__(self,
                    study_mode=False,
                    model_file='model_28.h5',
                    new_model_file=None,
                    gammaReward=0.1,
                    alphaUpdateNet=0.1,
                    epsilonRandom=0.05,
                    decayRandom=0.95,
                    players=None,
                    max_history_len=1000,
                    ):
        super().__init__()
        self.stats = commons.PlayerStats()
        self.new_model_file = new_model_file
        self.gammaReward = gammaReward
        self.alphaUpdateNet = alphaUpdateNet
        self.decayRandom = decayRandom
        self.epsilonRandom = epsilonRandom
        self.study_mode = study_mode
        self.max_history_len = max_history_len

        # self.my_seat = 0

        if model_file is not None:
            self.model = load(model_file)
        else:
            self.model = commons.model1()



    def good_moves(self, valid_actions, my_stack):
        good_moves = []

        good_moves.append({'action' : 'fold', 'amount': 0, 'type': 0})
        good_moves.append({'action' : 'call', 'amount': valid_actions[1]['amount'], 'type': 1})

        if valid_actions[2]['amount']['min'] == -1:
            return good_moves

        raise_min, raise_max = valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']

        fix = lambda x: max(min(x, raise_max), raise_min)

        good_moves.append({'action' : 'raise', 'amount': fix(self.small_blind_amount * 2), 'type': 2})
        good_moves.append({'action' : 'raise', 'amount': fix(my_stack // 2), 'type': 3})
        good_moves.append({'action' : 'raise', 'amount': fix(my_stack), 'type': 4})

        return good_moves


    def turn2vec(self, state_vec, move):
        move_numb = move['type']
        move_amount = move['amount']
        # print(move_numb, move_amount)
        X = np.concatenate((np.array(state_vec), np.array([move_numb, move_amount])))
        return X.reshape((1,-1))


    def find_best_strat(self, valid_actions, cur_state_vec, my_stack):
        best_move, best_reward = None, -1500*9
        good_moves = self.good_moves(valid_actions, my_stack)

        if sps.bernoulli.rvs(p=self.epsilonRandom):
            ind = sps.randint.rvs(low=0,high=len(good_moves))
            move = good_moves[ind]
            reward = float(self.model.predict(self.turn2vec(cur_state_vec, move), batch_size=1))
            return move, reward

        for move in good_moves:
            # print(cur_state_vec, move)
            cur_reward = float(self.model.predict(self.turn2vec(cur_state_vec, move), batch_size=1))
            # print(cur_reward)
            if cur_reward > best_reward:
                best_move = move
                best_reward = cur_reward

        return best_move, best_reward


    def declare_action(self, valid_actions, hole_card, round_state):
        self.stats.update(hole_card, round_state)

        round_state_vec = self.stats.calc_fine_params(hole_card, round_state)

        my_stack = round_state['seats'][self.my_seat]['stack']

        best_move, best_reward = self.find_best_strat(valid_actions,
                                                        round_state_vec,
                                                        my_stack)

        action, amount = best_move['action'], best_move['amount']

        return action, amount




    def receive_game_start_message(self, game_info):
        # self.my_name = game_info['seats'][self.my_seat]['name']
        self.stats.init_player_names(game_info)
        self.player_num = game_info["player_num"]
        self.max_round = game_info["rule"]["max_round"]
        self.small_blind_amount = game_info["rule"]["small_blind_amount"]
        self.ante_amount = game_info["rule"]["ante"]
        # self.blind_structure = game_info["rule"]["blind_structure"]

        # self.emulator.set_blind_structure(blind_structure)
        i = 0
        for player in game_info['seats']:
            if player['uuid'] == self.uuid:
                self.my_seat = i
                break
            i += 1


        self.stats = commons.PlayerStats()
        self.stats.init_player_names(game_info)


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.start_round_stack = seats[self.my_seat]['stack']


    def receive_street_start_message(self, street, round_state):
        pass


    def receive_game_update_message(self, action, round_state):
        if self.new_model_file is not None and self.study_mode:
            self.model.save(self.new_model_file)


    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


if __name__ == '__main__':

    player = MergePlayer()

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
            player.set_uuid(data.get('uuid'))
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
