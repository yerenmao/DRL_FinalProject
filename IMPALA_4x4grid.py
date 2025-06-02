import os
import sys
from pathlib import Path

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sumo_rl

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    env_name = "4x4grid"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file="nets/4x4grid/4x4.net.xml",
                route_file="nets/4x4grid/4x4.rou.xml",
                out_csv_name="outputs/4x4grid/impala",
                use_gui=False,
                single_agent=False,
                num_seconds=300000,
            )
        ),
    )

    config = (
        ImpalaConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=64)
        .training(
            train_batch_size=128,
            lr=2e-5,
            gamma=0.95,
            # grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
        )
        .debugging(log_level="ERROR")
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        # .api_stack(
        #     enable_rl_module_and_learner=False,
        #     enable_env_runner_and_connector_v2=False
        # )
    )

    local_dir = str(Path("~/ray_results/" + env_name).expanduser())

    tune.run(
        "IMPALA",
        name="IMPALA",
        stop={"timesteps_total": 1000000},
        checkpoint_freq=10,
        local_dir=local_dir,
        config=config.to_dict(),
    )
