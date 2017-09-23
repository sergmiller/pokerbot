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

from pypokerengine.players import RandomPlayer



class RLPokerPlayer(BasePokerPlayer):
    def __init__(self, nnet,
                    gammaReward=0.1,
                    alphaUpdateNet=0.1,
                    epsilonRandom=0.05,
                    players=[RandomPlayer()] * 8,
                    nnet_file_for_save=None,
                    ):
        super().__init__()
        if nnet is None:
            raise Exception('Try to use some nnet!')
        self.nnet = deepcopy(nnet)
        self.gammaReward = gammaReward
        self.alphaUpdateNet = alphaUpdateNet
        self.epsilonRandom = epsilonRandom
        self.players = [self] + players
        self.players = [(self.players[i], i) for i in range(len(self.players))]
        self.history = {'moves': [], 'states': [], 'rewards': []}

    def process_current_data(self, hole_card, round_state):
        my_pos = round_state['next_player']
        n_players = len(round_state['seats'])
        community_card = round_state['community_card']
        self.history['states'].append(None)

    def declare_action(self, valid_actions, hole_card, round_state):
        cur_state = self.process_current_data(hole_card, round_state)

        best_move = None
        best_reward = -np.inf
        for move in self.good_moves(valid_actions):
            cur_reward = self.nnet.predict(state2vec(cur_state, move), batch_size=1)
            if cur_reward > best_reward:
                best_move = move
        if best_move[1] < 2:
            call_action_info = valid_actions[best[1]]
            action, amount = call_action_info["action"], call_action_info["amount"]
        if best_move[1] >= 2:
            if valid_actions[2]['min'] != -1:
                call_action_info = valid_actions[2]
                action, amount = call_action_info["action"], call_action_info["amount"]["min"] \
                    if best[1] else min(2 * call_action_info["amount"]["min"], call_action_info["amount"]["max"])
            else:
                best[1] = 1
                call_action_info = valid_actions[1]
                action, amount = call_action_info["action"], call_action_info["amount"]

        self.history['states'].append(cur_state)
        self.history['moves'].append(best[1])
        return action, amount

    def receive_game_start_message(self, game_info):
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

        self.players = np.random.shuffle(self.players)
        # Register algorithm of each player which used in the simulation.
        for player in self.players:
            self.emulator.register_player(player[0])

        for i in range(len(self.players)):
            if self.players[i][1] == 0:
                self.my_seat = i


    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if len(history['states']) > 10 and sps.bernoulli.rvs(p=self.alphaUpdateNet):
            ind = sps.randint.rvs(high=history['states'] - 1)
            self.learn_with_states(history['states'][ind])

    def state2vec(self, state):
        pass

    def learn_with_states(state):
            if state.street ==  PokerConstants.Street.FINISHED:
                state.reward = state.round_reward[self.my_seat]
            else:
                state.reward += self.gammaReward * self.get_best_simulation(state)
            self.nnet.fit(self.state2vec(state), state.reward, epochs=1, verbose=0, batch_size=1)
