import sys

import cli_args

import torch

from policyflow.policyflow_torch.env import EnvPoolWrapper
from policyflow.policyflow_torch.storage import ReplayBuffer
from policyflow.policyflow_torch.modules import (
    Network,
    GaussianNetwork,
)
from policyflow.policyflow_torch.agents import Exo, ExoCfgInstance
from policyflow.policyflow_torch.runners import GymRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # ---- 环境配置 ----
    # envpool 支持的 mujoco 任务: Humanoid-v4, Ant-v4, HalfCheetah-v4, Hopper-v4 等
    env_name = "Humanoid-v4"
    num_envs = 4
    wrapped_env = EnvPoolWrapper(env_name, num_envs=num_envs)

    obs_dict, _ = wrapped_env.reset()
    critic_observations_size = obs_dict["critic_observations"].shape[1]
    actor_observations_size = obs_dict["actor_observations"].shape[1]

    # ---- Runner 配置 ----
    runner_cfg = cli_args.get_runner_cfg()
    runner_cfg["log_dir"] = "run/exo/" + env_name
    runner_cfg["max_iterations"] = 40000
    runner_cfg["rollouts"] = 24

    replay_buffer = ReplayBuffer(
        memory_size=runner_cfg.get("rollouts", 64),
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    num_actions = wrapped_env.action_space.shape[0]

    # ---- Exo 模型配置 ----
    model_cfg_dict = cli_args.get_exo_models_cfg()
    model_cfg_dict["critic"].update(
        {"input_size": critic_observations_size, "output_size": 1}
    )
    model_cfg_dict["actor"].update(
        {
            "input_size": actor_observations_size,
            "output_size": num_actions,
        }
    )
    models = {
        "critic": Network(**model_cfg_dict["critic"]),
        "actor": GaussianNetwork(**model_cfg_dict["actor"]),
    }

    # ---- Exo Agent 配置 ----
    agent_cfg = ExoCfgInstance()
    agent_cfg.beta = 5.0           # 指数软裁剪锐度
    agent_cfg.kl_coeff = 0.8       # KL 散度正则化系数
    agent_cfg.entropy_loss_scale = 0.01  # 熵正则化系数
    agent_cfg.learning_rate = 2e-4
    agent_cfg.learning_epochs = 5
    agent_cfg.mini_batches = 4

    agent = Exo(
        models=models,
        replay_buffer=replay_buffer,
        device=wrapped_env.device,
        cfg=agent_cfg.__dict__,
    )

    agent.init_replay_buffer(
        critic_observation_size=critic_observations_size,
        actor_observation_size=actor_observations_size,
        action_size=num_actions,
    )

    # ---- 训练 ----
    runner = GymRunner(env=wrapped_env, agent=agent, cfg=runner_cfg)
    runner.train(return_epochs=100)

    # close the simulator
    wrapped_env.close()


if __name__ == "__main__":
    # run the main function
    main()
