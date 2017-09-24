# -*- coding: UTF-8 -*-

import sys #импортируем библиотеку sys
import json #импортируем библиотеку json

from pypokerengine.players import BasePokerPlayer #импортируем класс BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate #импортируем

NB_SIMULATION = 200



class MyPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer" Не забывайте сделать родительский класс "BasePokerPlayer"

    #  we define the logic to make an action through this method (so this method would be the core of your AI) мы определяем логику для совершения действия с помощью этого метода (так что этот метод будет ядром вашего ИИ)

    def declare_action(self, valid_actions, hole_card, round_state):

        fold_action_info, call_action_info, raise_action_info  = valid_actions[0], valid_actions[1], valid_actions[2]

        pot_size = round_state["pot"]["main"]["amount"]

        self.nb_active = len([player for player in round_state['seats'] if player['state'] != 'folded'])

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
            if self.nb_active >=7 and win_rate >= 0.29:
                action = "raise"
                amount = raise_action_info["amount"]["max"]
                if amount == -1 or amount == -4:
                    action = "call"
                    amount = call_action_info["amount"]
            elif self.nb_active >=3 and win_rate >= 0.37:
                action = "raise"
                amount = raise_action_info["amount"]["max"]
                if amount == -1 or amount == -4:
                    action = "call"
                    amount = call_action_info["amount"]
            elif self.nb_active >=1 and win_rate >= 0.75:
                action = "raise"
                amount = raise_action_info["amount"]["max"]
                if amount == -1 or amount == -4:
                    action = "call"
                    amount = call_action_info["amount"]
            # elif win_rate >= 0.19:
            #     action = "raise"
            #     amount = min(pot_size, 4 * raise_action_info["amount"]["min"])
            #     if amount == -1 or amount == -4:
            #         action = "call"
            #         amount = call_action_info["amount"]
            elif win_rate >= 0.16 and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
            else:
                action = "fold"
                amount = fold_action_info["amount"]
                if free_flop:
                    action = "call"
                    amount = call_action_info["amount"]

        # if street == "flop":
        if win_rate >= percent * 1.2:
            action = "raise"
            amount = raise_action_info["amount"]["max"]
            if amount == -1 or amount == -4:
                action = "call"
                amount = call_action_info["amount"]
        elif win_rate >= percent:
            action = "raise"
            amount = min(pot_size, 4 * raise_action_info["amount"]["min"])
            if amount == -1 or amount == -4:
                action = "call"
                amount = call_action_info["amount"]
        else:
            action = "fold"
            amount = fold_action_info["amount"]

        # if street == "turn":
        #     if win_rate >= percent:
        #         action = "raise"
        #         amount = raise_action_info["amount"]["max"]
        #         if amount == -1 or amount == -4:
        #             action = "call"
        #             amount = call_action_info["amount"]
        #     elif win_rate >= percent*0.9:
        #         action = "raise"
        #         amount = min(pot_size, 4 * raise_action_info["amount"]["min"])
        #         if amount == -1 or amount == -4:
        #             action = "call"
        #             amount = call_action_info["amount"]
        #     elif win_rate >= percent*0.8:
        #         action = "call"
        #         amount = call_action_info["amount"]
        #     else:
        #         action = "fold"
        #         amount = fold_action_info["amount"]
        #
        # if street == "river":
        #     if win_rate >= percent:
        #         action = "raise"
        #         amount = raise_action_info["amount"]["max"]
        #         if amount == -1 or amount == -4:
        #             action = "call"
        #             amount = call_action_info["amount"]
        #     elif win_rate >= percent*0.8:
        #         action = "raise"
        #         amount = min(pot_size, 4 * raise_action_info["amount"]["min"])
        #         if amount == -1 or amount == -4:
        #             action = "call"
        #             amount = call_action_info["amount"]
        #     elif win_rate >= percent*0.8:
        #         action = "call"
        #         amount = call_action_info["amount"]
        #     else:
        #         action = "fold"
        #         amount = fold_action_info["amount"]

        if action == "raise":
            amount = max(amount, raise_action_info["amount"]["min"])
            amount = min(amount, raise_action_info["amount"]["max"])

        if action == 'raise' and amount == -1:
            action = 'call'
            amount = call_action_info['amount']

        #print 'hole_card:'
        #print hole_card

        #print 'win_rate:'
        #print win_rate

        #print 'percent:'
        #print percent

        #print 'nb_player'
        #print self.nb_active

        return action, amount # action returned here is sent to the poker engine действие, возвращаемое сюда, отправляется в покерный движок

    def receive_game_start_message(self, game_info):
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
