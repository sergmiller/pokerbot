from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.players import BasePokerPlayer
from baseline_players import RandomPlayer
from bots.honest_player import HonestPlayer
from bots.honest_modified_player import HonestModifiedPlayer
from bots.bot_denis_2 import BotDenisPlayer
from bots.bot1 import KoffaPlayer
# from bots.bot_denis_4 import
from bots.forl_condition_bot_2 import ConditionPlayer
from bots.botDenis import GloomCha
from bots.forl_condition_bot import HonestStatPlayer
from bots.fish_player import FishPlayer
from bots.fold_player import FoldPlayer
from bots.bot_denis_3 import BotDenis3Player
from copy import deepcopy
import scipy.stats as sps
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bots.RLbot import RLPokerPlayer
from tqdm import tqdm


N = 100

RLBotPark = []
DefaultBotPark = [RandomPlayer(), HonestPlayer(), FishPlayer(),
           BotDenisPlayer(), HonestModifiedPlayer(), ConditionPlayer(),
           GloomCha(), KoffaPlayer(), HonestStatPlayer(),
          ]

rl_score = []


for i in range(N):
    in_model_file = None if i == 0 else 'model_{}.h5'.format(i)
    out_model_file = 'model_{}.h5'.format(i + 1)
    config = setup_config(max_round=50, initial_stack=1500, small_blind_amount=15)
    humans = list(np.random.choice(DefaultBotPark, replace=True, size=4 + max(0, 4 - len(RLBotPark))))
    rlbots = list(np.random.choice(RLBotPark, replace=True, size=min(4, len(RLBotPark)))) if len(RLBotPark) > 0 else []
    print(humans, rlbots)
    players = humans + rlbots + [None]
    np.random.shuffle(players)
    print(players)
    i = 0
    for player in players:
        i += 1
        if player is None:
            RLP =  RLPokerPlayer(
                study_mode=True,
                model_file=in_model_file,
                gammaReward=0.1,
                alphaUpdateNet=0.1,
                epsilonRandom=0.05,
                players=players,
                max_history_len=1000,
                new_model_file=out_model_file,
            )
            config.register_player(name='RL',algorithm=RLP)
        else:
            config.register_player(name=str(player) + str(i), algorithm=player)
    info = start_poker(config, verbose=0)
    RLP.study_mode = False
    for player_info in info:
        if player_info['name'] == 'RL':
            stack = player_info['stack']
            rl_score.append(stack)
            print(stack)
            break
