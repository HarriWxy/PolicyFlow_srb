# AGENTS.md

## Quick Start
- **Environment**: Conda `env_isaaclab` (IsaacLab/IsaacSim required).
- **Install**: `cd policyflow && pip install -e .`
- **Train**: `python scripts/isaaclab/train.py --task=<TASK>-PF --headless` (see examples below).

## Repository Structure
- `policyflow/policyflow_torch/` – Core RL library (agents, env, modules, runners, storage, utils).
- `scripts/isaaclab/` – IsaacLab training/evaluation scripts.
- `scripts/multigoal/` – MultiGoal environment scripts.
- `scripts/gym/` – Gym (PointMaze) environment scripts.

## Key Commands
```bash
# IsaacLab training (example: Anymal-D)
python scripts/isaaclab/train.py \
    --task=Isaac-Velocity-Flat-Anymal-D-PF \
    --headless \
    --log_dir=runs_pf/Anymal_d_policyflow \
    --actor_hidden_dims 128 128 128 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 128 128 128 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=24

# MultiGoal training
python scripts/multigoal/train.py

# Gym training (PointMaze)
python scripts/gym/train.py
```

## Environment Registration
IsaacLab environments are registered in `scripts/isaaclab/register_envs.py` with `-PF` suffix (e.g., `Isaac-Velocity-Flat-Anymal-D-PF`). To add custom environments, follow the pattern in that file.

## Configuration
- **CLI arguments**: Defined in `scripts/isaaclab/cli_args.py` (and equivalents for multigoal/gym).
- **Agent configs**: Per-task hyperparameters in `scripts/isaaclab/agent_cfg.py` (e.g., `FrankaArmCfg`, `AnymalDCfg`).
- **Runner defaults**: `max_iterations=40000`, `rollouts=24`, `save_interval=100`, `log_dir=runs/policyflow`.

## Important Notes
- **No tests, linting, or CI**: This is a research repository; verify changes manually via training runs.
- **Headless mode**: Use `--headless` for faster training without rendering.
- **MultiGoal play**: Before running `scripts/multigoal/play.py`, update line 177 with your trained model path.
- **Dependencies**: `torch`/`torchvision` are expected from the IsaacLab conda environment (commented out in setup.py).

## References
- IsaacLab installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- Benchmark script: `scripts/isaaclab/run_benchmark.sh`
- Environment registration example: `scripts/isaaclab/register_envs.py`