import os

import cli_args

import torch
import envpool


from policyflow.policyflow_torch.env import GymEnvWrapper
from policyflow.policyflow_torch.storage import ReplayBuffer
from policyflow.policyflow_torch.modules import (
    Network,
    ContinuousNormalizingFlow,
    ConditionLinearLayer,
    FlowMlp,
)
from policyflow.policyflow_torch.agents import PolicyFlow, PolicyFlowCfgInstance
from policyflow.policyflow_torch.runners import GymRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    wrapped_env = GymEnvWrapper("Humanoid-v5", 128)

    obs_dict, _ = wrapped_env.reset()
    critic_observations_size = obs_dict["critic_observations"].shape[1]
    actor_observations_size = obs_dict["actor_observations"].shape[1]

    runner_cfg = cli_args.get_runner_cfg()
    replay_buffer = ReplayBuffer(
        memory_size=runner_cfg.get("rollouts", 64),
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )

    model_cfg_dict = cli_args.get_models_cfg()
    model_cfg_dict["critic"].update(
        {"input_size": critic_observations_size, "output_size": 1}
    )
    num_actions = wrapped_env.action_space.shape[1]

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

    runner = GymRunner(env=wrapped_env, agent=agent, cfg=runner_cfg)
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        current_path,
        "../..",
        "runs/policyflow/*/*.pt",
    )
    runner.load(model_path)
    runner.evaluate(steps=10000, return_epochs=100)

    # close the simulator
    wrapped_env.close()


if __name__ == "__main__":
    # run the main function
    main()
