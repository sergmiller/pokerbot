import sys
import json

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

## говнокод Дениса

FR_preflop_hands, SH_preflop_hands, HU_preflop_hands = [], [], [],
cards_power = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
suit = ['H','D','C','S']

pairs_from_two, pairs_from_eight, pairs_from_ten, pairs_from_jack, pairs_from_queen, pairs_from_king = [], [], [], [], [], []
premium_connectors_AK = []
premium_connectors_AQ = []
suited_connectors_from_JT = []

for i in cards_power[0:]:
    for j in range(4):
        pairs_from_two.append([suit[j-1]+i, suit[j]+i])
        pairs_from_two.append([suit[j]+i, suit[j-1]+i])

for i in cards_power[6:]:
    for j in range(4):
        pairs_from_eight.append([suit[j-1]+i, suit[j]+i])
        pairs_from_eight.append([suit[j]+i, suit[j-1]+i])

for i in cards_power[8:]:
    for j in range(4):
        pairs_from_ten.append([suit[j-1]+i, suit[j]+i])
        pairs_from_ten.append([suit[j]+i, suit[j-1]+i])

for i in cards_power[9:]:
    for j in range(4):
        pairs_from_jack.append([suit[j-1]+i, suit[j]+i])
        pairs_from_jack.append([suit[j]+i, suit[j-1]+i])

for i in cards_power[10:]:
    for j in range(4):
        pairs_from_queen.append([suit[j-1]+i, suit[j]+i])
        pairs_from_queen.append([suit[j]+i, suit[j-1]+i])

for i in cards_power[11:]:
    for j in range(4):
        pairs_from_king.append([suit[j-1]+i, suit[j]+i])
        pairs_from_king.append([suit[j]+i, suit[j-1]+i])

a = cards_power[-1]
k = cards_power[-2]
q = cards_power[-3]
j = cards_power[-4]
t = cards_power[-5]

for i in range(4):
    m = suit[i]
    mn = suit[i-1]
    premium_connectors_AK.append([mn+a, m+k])
    premium_connectors_AK.append([m+k, mn+a])
    premium_connectors_AK.append([m+a, m+k])
    premium_connectors_AK.append([m+k, m+a])

for i in range(4):
    m = suit[i]
    mn = suit[i-1]
    premium_connectors_AQ.append([mn+a, m+q])
    premium_connectors_AQ.append([m+q, mn+a])
    premium_connectors_AQ.append([m+a, m+q])
    premium_connectors_AQ.append([m+q, m+a])

for i in range(4):
    m = suit[i]
    suited_connectors_from_JT.append([m+q, m+j])
    suited_connectors_from_JT.append([m+j, m+t])
    suited_connectors_from_JT.append([m+j, m+q])
    suited_connectors_from_JT.append([m+t, m+j])

premium_connectors_all = premium_connectors_AQ + premium_connectors_AK

FR_preflop_hands = pairs_from_eight + premium_connectors_all
SH_preflop_hands = pairs_from_two + premium_connectors_all + suited_connectors_from_JT

## говнокод Дениса

NB_SIMULATION = 1000

class MyPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):

        def allin(action, amount):
            action = "raise"
            amount = raise_action_info["amount"]["max"]
            if amount == -1:
                action = "call"
                amount = call_action_info["amount"]
            return action, amount

        fold_action_info, call_action_info, raise_action_info  = valid_actions[0], valid_actions[1], valid_actions[2]

        pot_size = round_state["pot"]["main"]["amount"]

        self.nb_active = len([player for player in round_state['seats'] if player['state'] != 'folded' and player["state"] != "allin" and player["stack"]!= 0])

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
            if self.nb_active <= 2:
                if win_rate >= 0.75:
                    allin(action, amount)
                if win_rate >= 0.50 and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

            elif self.nb_active <= 6:
                if hole_card in pairs_from_king:
                    allin(action, amount)
                elif hole_card in pairs_from_jack or hole_card in premium_connectors_AK:
                    action = "raise"
                    amount = raise_action_info["amount"]["min"] * 2.5
                elif hole_card in SH_preflop_hands and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

            elif self.nb_active <= 9:
                if hole_card in pairs_from_king:
                    allin(action, amount)
                elif hole_card in premium_connectors_AK and call_action_info["amount"] == 30:
                    action = "raise"
                    amount = raise_action_info["amount"]["min"] * 2.5
                elif hole_card in FR_preflop_hands and call_action_info["amount"] == 30:
                    action = "call"
                    amount = call_action_info["amount"]
                else:
                    action = "fold"
                    amount = fold_action_info["amount"]

        else: #all streets
            if win_rate >= percent*1.1:
                allin(action, amount)
            elif win_rate >= percent*0.9 and call_action_info["amount"] <= 30:
                action = "call"
                amount = call_action_info["amount"]
            else:
                    action = "fold"
                    amount = fold_action_info["amount"]

        if action == "raise":
            amount = max(amount, raise_action_info["amount"]["min"])
            amount = min(amount, raise_action_info["amount"]["max"])

        return action, amount

    def receive_game_start_message(self, game_info):
        self.nb_players = game_info["player_num"]
        for (i, seat) in enumerate(game_info["seats"]):
            if seat["uuid"] == self.uuid:
                self.seat = i

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.nb_players = 0
        for i in range(len(seats)):
            if seats[i]["stack"] >= 30:
                self.nb_players += 1


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
