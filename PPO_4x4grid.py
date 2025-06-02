import os
import sys
from pathlib import Path

# —— 1. 設定 SUMO_HOME 路徑 —— 
os.environ.setdefault("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
raw = os.environ.get("SUMO_HOME", "")
raw = raw.strip('"').strip("'")
sumo_home = Path(raw)
if not sumo_home.exists():
    sys.exit(f"Error: SUMO_HOME={sumo_home} 不存在！請至系統環境變數確認。")

tools_dir = sumo_home / "tools"
if not tools_dir.is_dir():
    sys.exit(f"Error: {tools_dir} 不是目錄，請確認 SUMO_HOME 指向的是 Sumo 根目錄。")

sys.path.append(str(tools_dir))

# —— 2. 引入模組 —— 
import numpy as np
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sumo_rl

# —— 3. 建立環境 —— 
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    env_name = "4x4grid"

    def env_creator(_):
        net_abs = os.path.abspath(r"C:\Users\88692\sumo-rl\sumo_rl\nets\4x4-Lucas\4x4.net.xml")
        rou_abs = os.path.abspath(r"C:\Users\88692\sumo-rl\sumo_rl\nets\4x4-Lucas\4x4random.rou.xml")
        print("[RolloutWorker] Using net file:", net_abs)
        print("[RolloutWorker] Using route file:", rou_abs)
        return ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=net_abs,
                route_file=rou_abs,
                out_csv_name="outputs/4x4grid/ppo2",
                use_gui=True,              # ⚠️ 訓練時建議設為 False（加速）
                single_agent=False,
                num_seconds=80000,        # ✅ 一次 episode 跑滿 100000 steps
                          # ✅ 每 1000 steps 儲存一次 CSV
            )
        )

    register_env(env_name, env_creator)

    # —— 4. 設定 PPO Config —— 
    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # —— 5. 執行訓練 —— 
    local_dir = str(Path("~/ray_results2/" + env_name).expanduser())

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 80000},   # ✅ 總訓練步數：100000（剛好一個 episode）
        checkpoint_freq=10,
        local_dir=local_dir,
        config=config.to_dict(),
    )

