import sys
import json
import pprint
try:
    import commons
except:
    from . import commons

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

from collections import deque

from pypokerengine.players import RandomPlayer

from keras.models import save_model, load_model



class RLPokerPlayer(BasePokerPlayer):
    def __init__(self,
                    study_mode=False,
                    model_file=None,
                    gammaReward=0.1,
                    alphaUpdateNet=0.1,
                    epsilonRandom=0.05,
                    players=[RandomPlayer()] * 8 + [None],
                    n_games=10,
                    max_history_len=1000,
                    new_model_file=None,
                    ):
        super().__init__()
        self.stats = commons.PlayerStats()
        self.new_weights_file = new_weights_file
        self.gammaReward = gammaReward
        self.alphaUpdateNet = alphaUpdateNet
        self.epsilonRandom = epsilonRandom

        # self.my_seat = 0
        if study_mode:
            self.players = players
            for i in nrange(len(self.players)):
                if self.players[i] is None:
                    self.my_seat = i

        self.history = deque()

        if model_file is not None:
            self.model = load_model(model_file)
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


    def turn2vec(state_vec, move):
        move_numb = move['type']
        move_amount = move['amount']
        X = np.concatenate((state_vec, np.array([move_numb, move_amount])))
        return X


    def find_best_strat(self, valid_actions, cur_state_vec, my_stack):
        best_move, best_reward = None, -np.inf
        for move in self.good_moves(valid_actions, my_stack):
            cur_reward = self.model.predict(self.turn2vec(cur_state_vec, move), batch_size=1)
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

        if self.study_mode:
            self.update_history(round_state_vec, best_move, best_reward, round_state)
        return action, amount


    def update_history(self, round_state_vec, best_move, best_reward, round_state):
        cur_data_dict = {   'states_vec': round_state_vec,
                            'states_original': round_state,
                            'moves': best_move,
                            'rewards': best_reward,
                        }

        self.history.append(cur_data_dict)
        if len(self.history) == self.max_history_len:
            self.history.popleft()


    def receive_game_start_message(self, game_info):
        # self.my_name = game_info['seats'][self.my_seat]['name']
        self.stats.init_player_names(game_info)
        self.player_num = game_info["player_num"]
        self.max_round = game_info["rule"]["max_round"]
        self.small_blind_amount = game_info["rule"]["small_blind_amount"]
        self.ante_amount = game_info["rule"]["ante"]
        self.blind_structure = game_info["rule"]["blind_structure"]

        self.emulator = Emulator()
        self.emulator.set_game_rule(self.player_num,
                                    self.max_round,
                                    self.small_blind_amount,
                                    self.ante_amount)
        self.emulator.set_blind_structure(blind_structure)

        # Register algorithm of each player which used in the simulation.
        for player in self.players:
            self.emulator.register_player(player if player is not None else self)


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.start_round_stack = seats[self.my_seat]['stack']

    def receive_street_start_message(self, street, round_state):
        pass


    def receive_game_update_message(self, action, round_state):
        if self.new_model_file is not None:
            self.model.save_model(self.new_model_file)


    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.study_mode:
            best_move = {'action' : 'finish', 'amount': 0, 'type': 5}
            round_state_vec = -np.ones_like(history[0]['states_vec'])
            best_reward = round_state[self.my_seat] - self.start_round_stack

            self.update_history(round_state_vec, best_move, best_reward, round_state)


            if len(history['states']) > 10 and sps.bernoulli.rvs(p=self.alphaUpdateNet):
                ind = sps.randint.rvs(high=history['states'] - 1)
                self.learn_with_states(self.history[ind], self.history[ind+1])


    def do_best_simulation(next_state):
        possible_actions = self.emulator.generate_possible_actions(self, next_state)
        good_actions = self.good_actions(possible_actions, next_state['seats'][self.my_seat])
        best_reward = -np.inf
        for action in good_actions:
            next_next_state = self.emulator.apply_action(next_state, action['action'], action['amount'])
            next_next_actions = self.emulator.generate_possible_actions(self, next_next_state)
            best_reward = max(best_reward, self.find_best_strat(next_next_actions, next_next_state))
        return best_reward


    def learn_with_states(state,
                            next_state):
        if next_state['moves']['type'] == 6:
            reward = next_state['rewards']
        else:
            reward = next_state['rewards'] +\
            self.gammaReward * self.do_best_simulation(next_state['states'])

        self.model.fit(self.turn2vec(state['states_vec'], move),
                                reward, epochs=1, verbose=0, batch_size=1)
