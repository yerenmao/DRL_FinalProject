import argparse
import os
import sys

import pandas as pd
import pickle

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


env = SumoEnvironment(
    net_file="/Users/len/Desktop/DRL/final/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4_fixed5s.net.xml",
    route_file="/Users/len/Desktop/DRL/final/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
    use_gui=True,
    num_seconds=10000,
    min_green=5,
    delta_time=5,
)

done = {"__all__": False}
env.reset()
while not done["__all__"]:
    _, _, done, _ = env.step(action=None)

env.save_csv("outputs/4x4grid/ql/ql_multi/baseline.csv", 0)

env.close()
