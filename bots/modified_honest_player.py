from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import _montecarlo_simulation
from joblib import Parallel, delayed
import numpy as np

NB_SIMULATION = 1000

class ModifiedHonestPlayer(BasePokerPlayer):

    def __init__(self, nb_simulation=NB_SIMULATION):
        self.nb_simulation = nb_simulation

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = self.__parallel_estimate_hole_card_win_rate(
                nb_simulation=self.nb_simulation,
                nb_player=self.nb_player,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
        if win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]  # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info
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


    def __parallel_estimate_hole_card_win_rate(
                                    self,
                                    nb_simulation,
                                    nb_player,
                                    hole_card,
                                    community_card=None,
                                    ):
        if not community_card:
            community_card = []

        with Parallel(n_jobs=4) as parallel:
            win_count = np.sum(parallel(delayed(_montecarlo_simulation)
            (nb_player, hole_card, community_card) for _ in range(nb_simulation)))
        return 1.0 * win_count / nb_simulation
