import sys

import cli_args

import torch
# import envpool

from policyflow.policyflow_torch.env import EnvPoolWrapper
from policyflow.policyflow_torch.storage import ReplayBuffer
from policyflow.policyflow_torch.modules import (
    Network,
    ContinuousNormalizingFlow,
    ConditionLinearLayer,
    FlowMlp,
    GaussianNetwork,
)
from policyflow.policyflow_torch.agents import PolicyFlow, PolicyFlowCfgInstance, PPO, PPOCfgInstance, PolicyFlowOneStep
from policyflow.policyflow_torch.runners import GymRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    alg_name = "policyflow"  # or "ppo" policyflow
    env_name = "MetaWorld/Hammer-v3" # Ant-v5
    wrapped_env = EnvPoolWrapper(env_name, num_envs=2048)

    obs_dict, _ = wrapped_env.reset()
    critic_observations_size = obs_dict["critic_observations"].shape[1]
    actor_observations_size = obs_dict["actor_observations"].shape[1]

    runner_cfg = cli_args.get_runner_cfg()
    runner_cfg["log_dir"] = "run/" + alg_name + "/" + env_name
    replay_buffer = ReplayBuffer(
        memory_size=runner_cfg.get("rollouts", 64),
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    num_actions = wrapped_env.single_action_space.shape[0]
    
    if alg_name == "policyflow":
        model_cfg_dict = cli_args.get_policyflow_models_cfg()
        model_cfg_dict["critic"].update(
            {"input_size": critic_observations_size, "output_size": 1}
        )

        model_cfg_dict["actor"].update({"x_dim": num_actions})
        model_cfg_dict["actor"].update({"emb_dim": 32})

        nn_flow = FlowMlp(**model_cfg_dict["actor"]).to(wrapped_env.device)
        nn_condition = ConditionLinearLayer(
            cond_dim=actor_observations_size, emb_dim=32
        ).to(wrapped_env.device)

        models = {
            "critic": Network(**model_cfg_dict["critic"]),
            "actor": ContinuousNormalizingFlow(
                x_dims=num_actions,
                nn_flow=nn_flow,
                nn_condition=nn_condition,
                sample_steps=model_cfg_dict["flow_sample_steps"],
                interpolation_type="rectified_flow",
                device=wrapped_env.device,
            ),
        }

        agent_cfg = PolicyFlowCfgInstance()
        agent_cfg.gaussian_entropy_loss_scale = 0.01
        agent_cfg.brownian_reg_loss_scale = 0.0

        agent = PolicyFlowOneStep(
            models=models,
            replay_buffer=replay_buffer,
            device=wrapped_env.device,
            cfg=agent_cfg.__dict__,
        )
        
    elif alg_name == "ppo":
        model_cfg_dict = cli_args.get_ppo_models_cfg()
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

        agent_cfg = PPOCfgInstance()
        agent_cfg.entropy_loss_scale = 0.01
        
        agent = PPO(
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

    runner = GymRunner(env=wrapped_env, agent=agent, cfg=runner_cfg)
    runner.train(return_epochs=100)

    # close the simulator
    wrapped_env.close()


if __name__ == "__main__":
    # run the main function
    main()
