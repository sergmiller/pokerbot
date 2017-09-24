import sys
import json

import scipy.stats as sps

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


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

# говнокод Дениса

FR_preflop_hands, SH_preflop_hands, HU_preflop_hands = [], [], [],
cards_power = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suit = ['H', 'D', 'C', 'S']

pairs_from_two, pairs_from_eight, pairs_from_ten, pairs_from_jack, pairs_from_queen, pairs_from_king = [], [], [], [], [], []
premium_connectors_AK = []
premium_connectors_AQ = []
suited_connectors_from_JT = []

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



for i in cards_power[0:]:
    for j in range(4):
        pairs_from_two.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_two.append([suit[j] + i, suit[j - 1] + i])

for i in cards_power[6:]:
    for j in range(4):
        pairs_from_eight.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_eight.append([suit[j] + i, suit[j - 1] + i])

for i in cards_power[8:]:
    for j in range(4):
        pairs_from_ten.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_ten.append([suit[j] + i, suit[j - 1] + i])

for i in cards_power[9:]:
    for j in range(4):
        pairs_from_jack.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_jack.append([suit[j] + i, suit[j - 1] + i])

for i in cards_power[10:]:
    for j in range(4):
        pairs_from_queen.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_queen.append([suit[j] + i, suit[j - 1] + i])

for i in cards_power[11:]:
    for j in range(4):
        pairs_from_king.append([suit[j - 1] + i, suit[j] + i])
        pairs_from_king.append([suit[j] + i, suit[j - 1] + i])

a = cards_power[-1]
k = cards_power[-2]
q = cards_power[-3]
j = cards_power[-4]
t = cards_power[-5]

for i in range(4):
    m = suit[i]
    mn = suit[i - 1]
    premium_connectors_AK.append([mn + a, m + k])
    premium_connectors_AK.append([m + k, mn + a])
    premium_connectors_AK.append([m + a, m + k])
    premium_connectors_AK.append([m + k, m + a])

for i in range(4):
    m = suit[i]
    mn = suit[i - 1]
    premium_connectors_AQ.append([mn + a, m + q])
    premium_connectors_AQ.append([m + q, mn + a])
    premium_connectors_AQ.append([m + a, m + q])
    premium_connectors_AQ.append([m + q, m + a])

for i in range(4):
    m = suit[i]
    suited_connectors_from_JT.append([m + q, m + j])
    suited_connectors_from_JT.append([m + j, m + t])
    suited_connectors_from_JT.append([m + j, m + q])
    suited_connectors_from_JT.append([m + t, m + j])

premium_connectors_all = premium_connectors_AQ + premium_connectors_AK

FR_preflop_hands = pairs_from_eight + premium_connectors_all
SH_preflop_hands = pairs_from_two + premium_connectors_all + suited_connectors_from_JT

## говнокод Дениса

NB_SIMULATION = 500


class MyPlayer(BasePokerPlayer):
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
                    p=0.05,
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
        self.p = p

        # self.my_seat = 0

        if model_file is not None:
            self.model = load(model_file)
        else:
            self.model = commons.model1()

    def allin(self, raise_action_info, call_action_info):
        action = "raise"
        amount = raise_action_info["amount"]["max"]
        if amount == -1:
            action = "call"
            amount = call_action_info["amount"]

        return action, amount

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

        if sps.bernoulli.rvs(self.p):
            self.stats.update(hole_card, round_state)

            round_state_vec = self.stats.calc_fine_params(hole_card, round_state)

            my_stack = round_state['seats'][self.my_seat]['stack']

            best_move, best_reward = self.find_best_strat(valid_actions,
                                                            round_state_vec,
                                                            my_stack)

            action, amount = best_move['action'], best_move['amount']

            return action, amount


        fold_action_info, call_action_info, raise_action_info = valid_actions[0], valid_actions[1], valid_actions[2]

        pot_size = round_state["pot"]["main"]["amount"]

        self.nb_active = len([player for player in round_state['seats'] if
                              player['state'] != 'folded' and player["state"] == "allin" and player["stack"] != 0])

        if self.nb_active <= 1:
            self.nb_active = 2

        community_card = round_state['community_card']

        free_flop = self.seat == round_state["big_blind_pos"] and call_action_info["amount"] == 30

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_active,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        street = round_state["street"]
        percent = 1.0 / self.nb_active

        if street == "preflop":
            if self.nb_active <= 2:
                if win_rate >= 0.75:
                    return self.allin(raise_action_info, call_action_info)
                if win_rate >= 0.50 and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

            elif self.nb_active <= 6:
                if hole_card in pairs_from_king:
                    return self.allin(raise_action_info, call_action_info)
                elif hole_card in pairs_from_jack or hole_card in premium_connectors_all:
                    action = "raise"
                    amount = raise_action_info["amount"]["min"] * 2.5
                elif hole_card in SH_preflop_hands and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

            elif self.nb_active <= 9:
                if hole_card in pairs_from_king:
                    return self.allin(raise_action_info, call_action_info)
                elif hole_card in premium_connectors_AK and call_action_info["amount"] == 30:
                    action = "raise"
                    amount = raise_action_info["amount"]["min"] * 2.5
                elif hole_card in FR_preflop_hands and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

            if action == "fold" and free_flop and call_action_info["amount"] == 30:
                action = "call"
                amount = call_action_info["amount"]

        else:  # all streets
            if win_rate >= percent * 1.1:
                return self.allin(raise_action_info, call_action_info)
            elif win_rate >= percent * 0.9 and call_action_info["amount"] <= 30:
                action = "call"
                amount = call_action_info["amount"]
            else:
                action = "fold"
                amount = fold_action_info["amount"]

        if action == "raise":
            amount = max(amount, raise_action_info["amount"]["min"])
            amount = min(amount, raise_action_info["amount"]["max"])

        amount = int(amount)
        return action, amount

    def receive_game_start_message(self, game_info):
        self.nb_players = game_info["player_num"]
        for (i, seat) in enumerate(game_info["seats"]):
            if seat["uuid"] == self.uuid:
                self.seat = i

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
        self.nb_players = 0
        for i in range(len(seats)):
            if seats[i]["stack"] >= 30:
                self.nb_players += 1
        self.start_round_stack = seats[self.my_seat]['stack']

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        if self.new_model_file is not None and self.study_mode:
            self.model.save(self.new_model_file)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


if __name__ == '__main__':

    player = MyPlayer()

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
