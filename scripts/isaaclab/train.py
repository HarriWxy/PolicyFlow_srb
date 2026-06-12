import sys

import cli_args

from isaaclab.app import AppLauncher

parser = cli_args.get_parser()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
import random

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from policyflow_torch.env import IsaacLabEnvWrapper
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.modules import (
    Network,
    ContinuousNormalizingFlow,
    ConditionLinearLayer,
    FlowMlp,
)
from policyflow_torch.agents import PolicyFlow, PolicyFlowCfg
from policyflow_torch.runners import IsaaclabRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import register_envs
import register_srb_envs

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "pf_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: PolicyFlowCfg,
):
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = random.randint(0, 10000)
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("runs", "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for PolicyFlow
    wrapped_env = IsaacLabEnvWrapper(
        env=env, using_historical_obs=False, critic_obs_len=3, actor_obs_len=10
    )
    obs_dict, _ = wrapped_env.reset()
    critic_observations_size = obs_dict["critic_observations"].shape[1]
    actor_observations_size = obs_dict["actor_observations"].shape[1]

    runner_cfg = cli_args.get_runner_cfg()
    runner_cfg = cli_args.override_runner_cfg(runner_cfg, args_cli)
    replay_buffer = ReplayBuffer(
        memory_size=runner_cfg.get("rollouts", 64),
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )

    model_cfg_dict = cli_args.get_models_cfg()
    model_cfg_dict = cli_args.override_model_cfg(model_cfg_dict, args_cli)
    model_cfg_dict["critic"].update(
        {"input_size": critic_observations_size, "output_size": 1}
    )
    if hasattr(env.unwrapped, "action_manager"):
        num_actions = env.unwrapped.action_manager.total_action_dim
    else:
        num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)

    model_cfg_dict["actor"].update({"x_dim": num_actions})
    model_cfg_dict["actor"].update({"emb_dim": args_cli.obs_embeding_dims})

    nn_flow = FlowMlp(**model_cfg_dict["actor"]).to(wrapped_env.device)
    nn_condition = ConditionLinearLayer(
        cond_dim=actor_observations_size, emb_dim=args_cli.obs_embeding_dims
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

    agent = PolicyFlow(
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

    runner_cfg["max_iterations"] = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else runner_cfg["max_iterations"]
    )
    runner = IsaaclabRunner(env=wrapped_env, agent=agent, cfg=runner_cfg)
    runner.train(return_epochs=100)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
