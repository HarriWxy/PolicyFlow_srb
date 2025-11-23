import cli_args

import torch, os

from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.modules import Network, ContinuousNormalizingFlow
from policyflow_torch.agents import PolicyFlow, PPO
from policyflow_torch.runners import MultiGoalRunner
from policyflow_torch.modules import (
    Network,
    ContinuousNormalizingFlow,
    ConditionLinearLayer,
    FlowMlp,
    GaussianNetwork,
)

from multigoal import MultiGoalEnv
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    alg_name = "policyflow"  # or "ppo"

    current_path = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(os.path.join(current_path, "multigoal_env_cfg.yaml"))
    config.num_envs = config.eval_num_envs
    config.random_init_episode_length = False
    env = MultiGoalEnv(config)

    # wrap around environment
    wrapped_env = env
    obs_dict, _ = wrapped_env.reset()
    critic_observations_size = obs_dict["critic_observations"].shape[1]
    actor_observations_size = obs_dict["actor_observations"].shape[1]

    runner_cfg = cli_args.get_runner_cfg()
    replay_buffer = ReplayBuffer(
        memory_size=runner_cfg.get("rollouts", 64),
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )

    num_actions = 2

    if alg_name == "policyflow":
        model_cfg_dict = cli_args.get_policyflow_models_cfg()
        model_cfg_dict["critic"].update(
            {"input_size": critic_observations_size, "output_size": 1}
        )
        model_cfg_dict["actor"].update({"x_dim": num_actions})
        model_cfg_dict["actor"].update({"emb_dim": 64})

        nn_flow = FlowMlp(**model_cfg_dict["actor"]).to(wrapped_env.device)
        nn_condition = ConditionLinearLayer(
            cond_dim=actor_observations_size, emb_dim=64
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

        agent_cfg = cli_args.get_policyflow_agent_cfg()
        agent = PolicyFlow(
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

        agent_cfg = cli_args.get_ppo_agent_cfg()
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

    runner = MultiGoalRunner(env=wrapped_env, agent=agent, cfg=runner_cfg)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 16})
    model_lists = [
        "model_0.pt",
        "model_1000.pt",
        "model_2000.pt",
        "model_3000.pt",
    ]
    fig = plt.figure(figsize=(4 * len(model_lists), 4))
    for idx, model in enumerate(model_lists):
        ax = fig.add_subplot(1, len(model_lists), idx + 1)
        model_path = os.path.join(
            current_path,
            "../..",
            "runs/multigoal/25-09-26_19-13-35-454014",
            model
        )
        runner.load(model_path)

        obs_dict_list, actions_list, resets_list = runner.evaluate(steps=100)

        obs_list = []
        for obs_dict in obs_dict_list:
            obs_list.append(obs_dict["actor_observations"])
        env.draw_path_only(
            ax,
            obs_list,
            resets_list,
            legend=(idx==0)
        )
        ax.set_title(f"Paths ({idx * 1000} itertions)")
    plt.tight_layout()
    plt.show()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
