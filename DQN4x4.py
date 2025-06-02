import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from agents import DQNAgent
from sumo_rl.exploration import EpsilonGreedy
from tqdm import tqdm

import torch
import numpy as np

if __name__ == "__main__":
    runs = 1         # 跑幾組實驗（每個 run 都重開環境）
    episodes = 1     # 每組實驗跑幾回合（episode）

    alpha = 0.001
    gamma = 0.99
    decay = 0.99

    # os.makedirs("outputs/4x4_dqn_long", exist_ok=True)
    # os.makedirs("models/4x4_dqn_long", exist_ok=True)

    for run in tqdm(range(1, runs + 1), desc="Runs"):
        env = SumoEnvironment(
            net_file="nets/4x4-Lucas/4x4.net.xml",
            route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
            use_gui=True,
            num_seconds=80000,
            min_green=5,
            delta_time=5,
        )

        initial_states = env.reset()
        dqn_agents = {
            ts: DQNAgent(
                id=ts,
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=1.0,
                epsilon_min=0.005,
                epsilon_decay=0.99995,
                policy_network_hidden_dims=(64, 64),
                replay_buffer_size=10000,
                batch_size=64,
                target_update_freq=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            for ts in env.ts_ids
        }

        for episode in (pbar := tqdm(range(1, episodes + 1), desc=f"Run {run}", leave=False)):
            if episode != 1:
                initial_states = env.reset()

            # 設定初始狀態給每個 agent
            for ts in env.ts_ids:
                dqn_agents[ts].state = initial_states[ts]

            done = {"__all__": False}
            step_count = 0

            while not done["__all__"]:
                actions = {ts: dqn_agents[ts].act() for ts in dqn_agents.keys()}
                next_states, rewards, done, _ = env.step(actions)

                for ts in dqn_agents.keys():
                    dqn_agents[ts].observe(
                        next_state=next_states[ts],
                        reward=rewards[ts],
                        done=done[ts]
                    )
                
                step_count += 1
                pbar.set_postfix({"step": step_count})

                if (step_count + 1) % 1000 == 0:
                    env.save_csv(f"dqn-4x4grid", episode)

            # 可選：儲存 agent 網路
            for ts in dqn_agents:
                dqn_agents[ts].save_model(f"models/4x4_dqn/dqn_{ts}_run{run}_ep{episode}.pt")

        env.close()
