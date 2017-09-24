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

from pypokerengine.players import RandomPlayer

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


class RLPokerPlayer(BasePokerPlayer):
    def __init__(self,
                    study_mode=False,
                    model_file=None,
                    new_model_file=None,
                    gammaReward=0.1,
                    alphaUpdateNet=0.1,
                    epsilonRandom=0.05,
                    decayRandom=0.5,
                    players=[RandomPlayer()] * 8 + [None],
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
        if self.study_mode:
            self.players = players
            for i in np.arange(len(self.players)):
                if self.players[i] is None:
                    self.my_seat = i

        self.history = deque()

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

        print('action {}, amount {}'.format(action, amount))

        if self.study_mode:
            self.update_history(round_state_vec, best_move, best_reward, round_state)
        return action, amount


    def update_history(self, round_state_vec, best_move, best_reward, round_state):
        cur_data_dict = {   'states_vec': round_state_vec,
                            'states_original': round_state,
                            'moves': best_move,
                            'rewards': best_reward,
                            'stats': deepcopy(self.stats),
                        }

        self.history.append(cur_data_dict)
        if len(self.history) == self.max_history_len:
            self.history.popleft()


    def receive_game_start_message(self, game_info):
        # self.my_name = game_info['seats'][self.my_seat]['name']
        if self.study_mode:
            self.uuid = game_info["seats"][self.my_seat]['uuid']
        self.stats.init_player_names(game_info)
        self.player_num = game_info["player_num"]
        self.max_round = game_info["rule"]["max_round"]
        self.small_blind_amount = game_info["rule"]["small_blind_amount"]
        self.ante_amount = game_info["rule"]["ante"]
        # self.blind_structure = game_info["rule"]["blind_structure"]

        self.emulator = Emulator()
        self.emulator.set_game_rule(self.player_num,
                                    self.max_round,
                                    self.small_blind_amount,
                                    self.ante_amount)
        # self.emulator.set_blind_structure(blind_structure)

        # Register algorithm of each player which used in the simulation.
        for i in np.arange(self.player_num):
            self.emulator.register_player(uuid=game_info["seats"][i]["uuid"],
            player=self.players[i] if self.players[i] is not None else self)


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.start_round_stack = seats[self.my_seat]['stack']


    def receive_street_start_message(self, street, round_state):
        pass


    def receive_game_update_message(self, action, round_state):
        if self.new_model_file is not None and self.study_mode:
            self.model.save(self.new_model_file)


    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.study_mode:
            best_move = {'action' : 'finish', 'amount': 0, 'type': 5}
            round_state_vec = -np.ones(commons.FEATURES_LEN - 2)
            best_reward = round_state['seats'][self.my_seat]['stack'] - self.start_round_stack

            self.update_history(round_state_vec, best_move, best_reward, round_state)


            if len(self.history) > 10 and sps.bernoulli.rvs(p=self.alphaUpdateNet):
                ind = sps.randint.rvs(low=0, high=len(self.history) - 2)
                self.learn_with_states(self.history[ind], self.history[ind+1])

            self.epsilonRandom *= self.decayRandom


    # def generate_possible_actions(self, game_state):
    #     players = game_state["seats"]
    #     player_pos = game_state["next_player"]
    #     sb_amount = game_state["small_blind_amount"]
    #     return ActionChecker.legal_actions(players, player_pos, sb_amount)

    def do_best_simulation(self, next_state, next_state_vec, next_state_stats):
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(next_state)
        # pp.pprint('*********************************************')
        game_state = restore_game_state(next_state)
        possible_actions = self.emulator.generate_possible_actions(game_state)
        good_actions = self.good_moves(possible_actions, next_state['seats'][self.my_seat]['stack'])
        best_reward = -1500*9
        for action in good_actions:
            # print(action)
            try:
                next_next_game_state = self.emulator.apply_action(game_state, action['action'], action['amount'])
                next_next_state = next_next_game_state[1][-1]['round_state']
                # next_next_state = next_next_game_state['action_histories']
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(next_next_state)
                if next_next_state['street'] in  ['showdown','finished',4,5]:
                    best_reward = max(best_reward, next_next_state['seats'][self.my_seat]['stack'] - self.start_round_stack)
                else:
                    next_next_game_state = restore_game_state(next_next_state)
                    hole_card = attach_hole_card_from_deck(next_next_game_state, self.uuid)
                    next_state_stats.update(hole_card, next_next_state)
                    next_next_state_vec = self.stats.calc_fine_params(hole_card, next_next_state)

                    next_next_actions = self.emulator.generate_possible_actions(next_next_game_state)
                    best_reward = max(best_reward, self.find_best_strat(next_next_actions,
                                                                        next_next_state_vec,
                                                                        next_next_state['seats'][self.my_seat]['stack'],
                                                                        )[1])
            except:
                continue
        return best_reward


    def learn_with_states(self, state,
                            next_state):
        if next_state['states_original']['street'] in ['showdown','finished',4,5]:
            reward = next_state['rewards']
        else:
            reward = next_state['rewards'] +\
            self.gammaReward * self.do_best_simulation( next_state['states_original'],
                                                        next_state['states_vec'],
                                                        next_state['stats'],
                                                    )
        X = self.turn2vec(state['states_vec'], state['moves'])
        y = np.array([reward])
        # print(X, y, X.shape, y.shape)
        self.model.fit(x=X,y=y, epochs=1, verbose=0, batch_size=1)
