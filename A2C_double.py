import os
import sys

import gymnasium as gym
from stable_baselines3 import A2C


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

os.makedirs("outputs/double", exist_ok=True)
os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="nets/double/network.net.xml",
        route_file="nets/double/flow.rou.xml",
        out_csv_name="outputs/double/a2c",
        single_agent=True,
        use_gui=False,
        num_seconds=100000,
    )

    model = A2C(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        # learning_starts=0,
        # train_freq=1,
        # target_update_interval=500,
        # exploration_initial_eps=0.05,
        # exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=100000)
    model.save("models/a2c_double")
