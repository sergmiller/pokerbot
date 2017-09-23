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
FEATURES_LEN = 14 * 9

from pypokerengine.players import BasePokerPlayer
from copy import deepcopy

STREET_NAMES = ['preflop', 'flop', 'turn', 'river']

class PlayerStats(object):
    def __init__(self):
        eps = 1e-3
        street_stats = {'total':{'fold':eps,'raise':eps,'call':eps},'count':{'fold':0,'raise':0,'call':0}}
        self.street_stats = dict()
        for street in ['preflop', 'flop', 'turn', 'river']:
            self.street_stats[street] = deepcopy(street_stats)
        self.player_stats = dict()

    def _init_player(self, name):
        self.player_stats[name] = deepcopy(self.street_stats)

    def init_player_names(self, game_info):
        for player_info in game_info['seats']:
            self._init_player(player_info['name'])

    # call in receive_game_update_message
    def update_on_action(self, action, round_state):
        '''
        Parameters:
            action: {'player_uuid': 'kvsetmwlrlfhctxosjhbdf', 'amount': 0, 'action': 'fold'}
            round_state
        '''
        # getting player name
        uuid = action['uuid']
        for seat in round_state['seats']:
            if seat['uuid'] == uuid:
                player_name = seat['name']
                break

        # updating counts
        street_stats = self.player_stats[player_name][round_state['street']]
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
        for seat in round_states['seats']:
            name = seat['name']
            seat_stats[name] = dict()
            seat_stats[name]['participating'] = 1 if seat['state'] != 'folded' else 0
            curr_history = round_states['action_histories'][round_states['street']]
            bid = 0
            for i in range(len(curr_history)-1, 0, -1):
                if seat['uuid'] == curr_history[i]['uuid']:
                    try:
                        bid = curr_history[i]['amount'] if 'amount' in curr_history[i] else 0
                    except:
                        print(curr_history[i])
                        print(curr_history)
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
        return params, stats

    def calc_fine_params(self, hole_card, round_states):
        self_seat_id = round_states['next_player']
        n_players = len(round_states['seats'])
        players_params, players_stats = self.calc_params(hole_card, round_states)
        fine_params = []
        params_per_name = len(players_params) / n_players
        for ind in range(1, n_players+1):
            i = (ind+1)%n_players
            for param in ['participating', 'confidence']:
                fine_params.append(players_stats[round_states['seats'][i]['name']][param])
            for param1 in ['preflop', 'flop', 'turn', 'river']:
                for param2 in ['fold','call','raise']:
                    fine_params.append(players_stats[round_states['seats'][i]['name']][param1][param2])
            #fine_params.extend(players_params[i*params_per_name:(i+1)*params_per_name])
        assert len(fine_params) == len(players_params)
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
from keras.activations import sigmoid
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, mean_squared_error
from keras.callbacks import LearningRateScheduler


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
    model.compile(loss=mean_squared_error,
                  optimizer=sgd,
                 )
    return model
