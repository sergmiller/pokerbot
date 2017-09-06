import numpy as np
# import time

from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator

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
