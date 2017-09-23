from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import scipy.stats as sps

NB_SIMULATION = 50


def norm(x):
    return max(min(x, 0), 1)


class HonestStatPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        fold_action_info = valid_actions[0]
        call_action_info = valid_actions[1]
        raise_action_info = valid_actions[2]

        call_amount = call_action_info["amount"]
        raise_amount_max = raise_action_info["amount"]["max"]
        raise_amount_min = raise_action_info["amount"]["min"]

        pot_size = round_state["pot"]["main"]["amount"]
        community_card = round_state['community_card']

        street_nums = {
            "preflop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
        }
        street_num = street_nums[round_state["street"]]

        nb_player = self.nb_player
        for seat in round_state["seats"]:
            if seat["state"] == "folded":
                nb_player -= 1

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        params = [
            norm(win_rate),
            norm(max(call_amount, 30) / abs(raise_amount_max)),
            norm(pot_size / 13500),
            street_num,
            norm(abs(raise_amount_max) / 13500)
        ]

        decision_function = 100 * params[0] + -86.2 * params[1] + 104.8 * params[2] + 27.5 * params[3] + 54.9 * params[
            4] + sps.norm.rvs(size=1, scale=10 ** .5)

        if decision_function > 100:
            action = raise_action_info["action"]
            amount = min(pot_size, 4 * raise_amount_min)
        elif 100 > decision_function > 80:
            action = raise_action_info["action"]
            amount = 2 * raise_amount_min
        elif 80 > decision_function > 40:
            action = call_action_info["action"]
            amount = call_action_info["amount"]
        elif decision_function < 40:
            action = fold_action_info["action"]
            amount = fold_action_info["amount"]
        else:
            action = fold_action_info["action"]
            amount = fold_action_info["amount"]

        return action, amount

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
