import sys
import json
import pprint
try:
    import commons
except:
    from . import commons

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        self.nb_active = len([player for player in round_state['seats'] if player['state'] != 'folded'])
        # print('****************************\nMY_INFO {}\n*********************************\n'\
        # .format(self.nb_active),
        # file=sys.stderr)
        # print(json.dumps(round_state, ensure_ascii=False))
        # pprint(json.dumps(round_state, ensure_ascii=False))
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(round_state)
        # print('FISH: hole card - {} converted - {}'.format(hole_card, [(x.suit, x.rank) for x in gen_cards(hole_card)]))

        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]
        # print(game_info)
        # print('FISH: num - {} max_round - {} sb - {} ante - {} bl_st - {}'.format(
        #     player_num, max_round, small_blind_amount, ante_amount, blind_structure
        # ))
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(game_info)
        # print(type(game_info))
        # print(game_info.get('uuid'))
        # print(self.uuid)


    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(action)
        # pp.pprint(round_state)
        # pp.pprint(round_state)
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(winners)
        # pp.pprint(hand_info)
        # pp.pprint(round_state)
        pass
        # print(winners)


if __name__ == '__main__':

    player = FishPlayer()

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
