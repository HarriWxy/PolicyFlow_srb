import gymnasium as gym

gym.register(
    id="Isaac-Lift-Cube-Franka-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:FrankaArmCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-Anymal-C-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg:NavigationEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:NavigationCfg",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-PF",
    entry_point="isaaclab_tasks.direct.quadcopter.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:QuadcopterCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg:AnymalDFlatEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:AnymalDCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:UnitreeGo2Cfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-H1-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg:H1RoughEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:UnitreeH1Cfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg:G1RoughEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:UnitreeG1Cfg",
    },
)

gym.register(
    id="Isaac-Open-Drawer-Franka-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg:FrankaCabinetEnvCfg",
        "pf_cfg_entry_point": "agent_cfg:CabinetCfg",
    },
)
