import sys
import json

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


class KoffaPlayer(BasePokerPlayer):
    def __init__(self, FOLD_RATE_PREFLOP = 0.60, FOLD_RATE = 0.70, RAISE_RATE = 0.80, PUSH_RATE = 0.90, NB_SIMULATION = 500):
        self.FOLD_RATE_PREFLOP = FOLD_RATE_PREFLOP
        self.FOLD_RATE = FOLD_RATE
        self.RAISE_RATE = RAISE_RATE
        self.PUSH_RATE = PUSH_RATE
        self.NB_SIMULATION = NB_SIMULATION

    def declare_action(self, valid_actions, hole_card, round_state):
        fold_action_info = valid_actions[0]
        call_action_info = valid_actions[1]
        raise_action_info = valid_actions[2]

        pot = round_state["pot"]["main"]["amount"]

        nb_player = 9
        for seat in round_state["seats"]:
            if seat["state"] == "folded":
                nb_player -= 1

        street = round_state["street"]

        if street == "preflop":
            nb_player = 2

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.NB_SIMULATION,
            nb_player=nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(round_state['community_card'])
        )

        # print(hole_card)
        # print(win_rate)

        call_amount = call_action_info["amount"]
        want_see_flop = street == "preflop" and call_amount <= self.bb * 2 and win_rate > self.FOLD_RATE_PREFLOP
        free_flop = self.seat == round_state["big_blind_pos"] and call_action_info["amount"] == self.bb

        if win_rate < self.FOLD_RATE:
            if call_action_info["amount"] == 0 or free_flop or want_see_flop:
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif win_rate < self.RAISE_RATE:
            if call_action_info["amount"] > self.bb * 3:
                action, amount = fold_action_info["action"], fold_action_info["amount"]
            else:
                action, amount = call_action_info["action"], call_action_info["amount"]
        elif win_rate >= self.RAISE_RATE:
            if call_action_info["amount"] > raise_action_info["amount"]['max'] * 0.20:
                action, amount = fold_action_info["action"], fold_action_info["amount"]
            elif raise_action_info["amount"]['min'] == -1:
                action, amount = call_action_info["action"], call_action_info["amount"]
            elif win_rate < self.PUSH_RATE:
                action, amount = raise_action_info["action"], raise_action_info["amount"]['min'] * 2
                if amount > self.bb * 6:
                    amount = self.bb * 6
                    if amount < raise_action_info["amount"]['min']:
                        action, amount = fold_action_info["action"], fold_action_info["amount"]
            else:
                action, amount = raise_action_info["action"], pot * 0.30
                if amount > raise_action_info["amount"]["max"]:
                    amount = raise_action_info["amount"]["max"]
                elif amount < raise_action_info["amount"]["min"]:
                    amount = raise_action_info["amount"]["min"]
        else:
            action, amount = fold_action_info["action"], fold_action_info["amount"]

        return action, amount

    def receive_game_start_message(self, game_info):
        self.sb = game_info["rule"]["small_blind_amount"]
        self.bb = self.sb * 2

        for (i, seat) in enumerate(game_info["seats"]):
            if seat["uuid"] == self.uuid:
                self.seat = i

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return KoffaPlayer()

if __name__ == '__main__':

    player = KoffaPlayer()

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
