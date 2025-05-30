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

os.makedirs("models/ql", exist_ok=True)

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 5

    env = SumoEnvironment(
        net_file="nets/2x2grid/2x2.net.xml",
        route_file="nets/2x2grid/2x2.rou.xml",
        use_gui=False,
        num_seconds=40000,
        min_green=5,
        delta_time=5,
    )

    initial_states = env.reset()
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
        )
        for ts in env.ts_ids
    }

    for episode in range(1, episodes + 1):
        if episode != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
            
            env.save_csv(f"outputs/2x2grid/ql/ql_multi", episode)

    for ts, agent in ql_agents.items():
        model_path = f"models/ql/ql_table_{ts}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(agent.q_table, f)
        print(f"Saved {ts} Q-table to {model_path}")
    env.close()
