import numpy as np
# import time

from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator

############################# MONTECARLO SIMULATION ############################

# def timer(f):
#     def wrapped(*args, **kwargs):
#         s = time.time()
#         res = f(*args, **kwargs)
#         print('worktime: {}'.format(time.time() - s))
#         return res
#     return wrapped

# @timer
def estimate_hole_card_win_rate(
                            nb_simulation,
                            nb_player,
                            hole_card,
                            community_card=None):
    if not community_card:
        community_card = []
    need_cards = 5 - len(community_card) + 2 * (nb_player - 1)
    possible_cards = (map(lambda id: Card.from_id(id + 1),
                                    np.random.choice(
                                    a=52,
                                    size=need_cards,
                                    replace=False,
                                ))
                                for _ in np.arange(nb_simulation))
    win_count = np.sum(montecarlo_simulation(
                                                    nb_player,
                                                    hole_card,
                                                    community_card,
                                                    list(next_cards))
                        for next_cards in possible_cards)
    return 1.0 * win_count / nb_simulation


def montecarlo_simulation(
                        nb_player,
                        hole_card,
                        community_card,
                        next_cards):
    len_to_com = 5 - len(community_card)
    community_card = community_card + next_cards[-len_to_com:]
    opponents_hole = [next_cards[2 * i : 2 * i + 2]
            for i in np.arange(nb_player - 1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card)
                                    for hole in opponents_hole]

    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

############################# FEATURES_EXTRACTION ##############################
FEATURES_LEN = 14 * 9 + 8
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from copy import deepcopy

STREET_NAMES = ['preflop', 'flop', 'turn', 'river']

class PlayerStats(object):
    def __init__(self, nb_simulations=100):
        self.nb_simulations = nb_simulations
        eps = 1e-3
        street_stats = {'total':{'fold':eps,'raise':eps,'call':eps},'count':{'fold':0,'raise':0,'call':0}}
        self.street_stats = dict()
        for street in ['preflop', 'flop', 'turn', 'river']:
            self.street_stats[street] = deepcopy(street_stats)
        self.player_stats = dict()

    def init_by_vec(self, vec, round_state):
        self.player_stats = dict()
        for seat_id in range(round_state['seats']):
            self._init_player(seat['uuid'])

    def _init_player(self, name):
        self.player_stats[name] = deepcopy(self.street_stats)

    def init_player_names(self, game_info):
        self.nb_players = game_info['player_num']
        for player_info in game_info['seats']:
            self._init_player(player_info['uuid'])

    # call in receive_game_update_message
    def update_on_action(self, action, round_state, uuid_name='uuid'):
        '''
        Parameters:
            action: {'player_uuid': 'kvsetmwlrlfhctxosjhbdf', 'amount': 0, 'action': 'fold'}
            round_state
        '''
        # getting player name
        uuid = action[uuid_name]
#         for seat in round_state['seats']:
#             if seat['uuid'] == uuid:
#                 player_name = seat['name']
#                 break

        # updating counts
        street_stats = self.player_stats[uuid][round_state['street']]
        if action['action'] != 'SMALLBLIND' and action['action'] != 'BIGBLIND':
            for act in street_stats['total']:
                street_stats['total'][act] += 1
            street_stats['count'][action['action'].lower()] += 1

    def update(self, hole_card, round_state):
        for stage in round_state['action_histories']:
            for action in round_state['action_histories'][stage]:
                self.update_on_action(action, round_state)

    # returns params
    def calc_params(self, hole_card, round_states):
        freq = 'freq'
        stats = dict()
        params = []
        seat_stats = dict()
        n_active_players = 0
        for seat in round_states['seats']:
            name = seat['uuid']
            seat_stats[name] = dict()
            seat_stats[name]['participating'] = 1 if seat['state'] != 'folded' else 0
            n_active_players += seat_stats[name]['participating']
            curr_history = round_states['action_histories'][round_states['street']]
            bid = 0
            for i in range(len(curr_history)-1, 0, -1):
                if seat['uuid'] == curr_history[i]['uuid']:
                    bid = curr_history[i]['amount'] if 'amount' in curr_history[i] else 0
            seat_stats[name]['confidence'] = bid / float(seat['stack'] + bid) if bid > 0 else 0

        # calculate frequencies for fold/call/raise in each stage of the game
        for player in self.player_stats:
            stats[player] = dict()
            for street in ['preflop', 'flop', 'turn', 'river']:
                stats[player][street] = dict()
                for stat in ['fold','call','raise']:
                    stats[player][street][stat] = self.player_stats[player][street]['count'][stat] / \
                        float(self.player_stats[player][street]['total'][stat])
                    params.append(stats[player][street][stat])
            for seat_stat in seat_stats[player]:
                stats[player][seat_stat] = seat_stats[player][seat_stat]
                params.append(seat_stats[player][seat_stat])

        return params, stats, n_active_players

    def calc_fine_params(self, hole_card, round_states):
        self_seat_id = round_states['next_player']
        n_players = len(round_states['seats'])
        players_params, players_stats, n_active_players = self.calc_params(hole_card, round_states)
        fine_params = []
        params_per_name = len(players_params) / n_players
        for ind in range(1, n_players+1):
            i = (ind+1)%n_players
            for param in ['participating', 'confidence']:
                fine_params.append(players_stats[round_states['seats'][i]['uuid']][param])
            for param1 in ['preflop', 'flop', 'turn', 'river']:
                for param2 in ['fold','call','raise']:
                    fine_params.append(players_stats[round_states['seats'][i]['uuid']][param1][param2])
            #fine_params.extend(players_params[i*params_per_name:(i+1)*params_per_name])
        assert len(fine_params) == len(players_params)

        global_params = []
        community_card = round_states['community_card']
        win_rate = estimate_hole_card_win_rate(
                    nb_simulation=self.nb_simulations,
                    nb_player=n_active_players,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
        global_params.append(win_rate)
        for i in range(self_seat_id - 1, -1, -1):
            ind = (i + n_players) % n_players

        player_stack = round_states['seats'][self_seat_id]['stack']
        gain_stack_relation = player_stack / float(round_states['pot']['main']['amount']) if player_stack > 0 else 10.
        global_params.append(gain_stack_relation)
        global_params.append(1. / n_active_players)
        global_params.append(round_states['round_count'] / 50.) # stage of the game
        global_params.append(len(round_states['action_histories'])) # stage of the round
        global_params.append(len(round_states['action_histories'][round_states['street']]) /\
                             float(n_players)) # number of actions in the street

        fine_params.extend(global_params)
        return fine_params

    def get_round_id(self, round_states):
        return round_states['round_count']


############################ KERAS MODEL #######################################

import h5py
import keras.backend as K


from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid, linear
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.losses import binary_crossentropy, mse
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score
from keras.callbacks import LearningRateScheduler
from keras import regularizers


def model1(input_shape=FEATURES_LEN):
    model = Sequential()
    model.add(Dense(units=32, input_shape=(input_shape,),
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(0.01)
                   ))

    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(units=16,
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(0.01)
                   ))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(0.01),
                    activation=linear,
                   ))

    sgd = SGD(lr=1e-2,
              momentum=0.9,
              decay=0.5,
              nesterov=True
             )
    model.compile(loss=mse,
                  optimizer=sgd,
                 )
    return model
