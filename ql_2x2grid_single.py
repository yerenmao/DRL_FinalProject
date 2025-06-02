import os
import sys
import numpy as np
import pickle
from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("請先設定環境變數 SUMO_HOME")
import traci

EPISODES = 5
MAX_STEPS = 100000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

env = SumoEnvironment(
    net_file="nets/2x2grid/2x2.net.xml",
    route_file="nets/2x2grid/2x2.rou.xml",
    out_csv_name="outputs/2x2grid/ql",
    single_agent=True,
    use_gui=False,
    num_seconds=40000,
    min_green=5,
    delta_time=5,
)

action_space = env.action_space
# print(env.action_space)

# Q-table: 狀態 (tuple) -> np.ndarray of action values
Q = {}

def get_state(obs):
    waits = obs[:4]
    buckets = tuple(np.clip((waits // 5).astype(int), 0, 4))
    return buckets 

for ep in range(1, EPISODES + 1):
    obs, _ = env.reset()
    state = get_state(obs)

    if state not in Q:
        Q[state] = np.zeros(action_space.n)

    total_reward = 0
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        if np.random.rand() < EPSILON:
            action = action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = get_state(next_obs)
        total_reward += reward

        if next_state not in Q:
            Q[next_state] = np.zeros(action_space.n)

        best_next = np.max(Q[next_state])
        Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

        state = next_state
        done = terminated or truncated
        step += 1

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    
    print(f"Episode {ep} 完成, 步數: {step}, 總 reward: {total_reward:.2f}, epsilon: {EPSILON:.4f}")
    env.save_csv(f"outputs/2x2grid/ql/ql_single", ep)


with open("models/ql_single_2x2grid.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Complete! Model saved to models/ql_single_2x2grid.pkl")

env.close()