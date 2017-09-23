import sys #импортируем библиотеку sys
import json #импортируем библиотеку json

from pypokerengine.players import BasePokerPlayer #импортируем класс BasePokerPlayer

premium_hands = []
lnn = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
suit = ['H','D','C','S']

premium_pairs = []
premium_connectors = []

for i in lnn[6:]:
    for j in range(4):
        premium_pairs.append([suit[j-1]+i, suit[j]+i])
        premium_pairs.append([suit[j]+i, suit[j-1]+i])

a = lnn[-1]
k = lnn[-2]
q = lnn[-3]

for i in range(4):
    m = suit[i]
    mn = suit[i-1]
    premium_connectors.append([mn+a, m+k])
    premium_connectors.append([m+k, mn+a])
    premium_connectors.append([m+a, m+k])
    premium_connectors.append([m+k, m+a])

premium_hands = premium_pairs + premium_hands

class GloomCha(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer" Не забывайте сделать родительский класс "BasePokerPlayer"

    #  we define the logic to make an action through this method (so this method would be the core of your AI) мы определяем логику для совершения действия с помощью этого метода (так что этот метод будет ядром вашего ИИ)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        if hole_card not in premium_hands:
            action = "fold"
            action_info = valid_actions[0]
            amount = action_info["amount"]
        if hole_card in premium_hands:
            action = "raise"
            action_info = valid_actions[2]
            amount = action_info["amount"]["max"]
            if amount == -1: action = "call"
            if action == "call":
                action_info = valid_actions[1]
                amount = action_info["amount"]
        return action, amount # action returned here is sent to the poker engine действие, возвращаемое сюда, отправляется в покерный движок

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


if __name__ == '__main__':

    player = GloomCha()

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

