from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import _montecarlo_simulation
from joblib import Parallel, delayed
import numpy as np
import bots.commons

NB_SIMULATION = 300

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



if __name__ == '__main__':

    player = ModifiedHonestPlayer()

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
