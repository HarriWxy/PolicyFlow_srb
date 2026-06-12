import gymnasium as gym

# Space Robotics Bench (SRB) environments registered for PolicyFlow
# These tasks inherit from IsaacLab's ManagerBasedRLEnv, making them compatible with PolicyFlow.

# Example: Debris Capture task
gym.register(
    id="srb/debris_capture-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.debris_capture.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:DebrisCaptureCfg",
    },
)

# Example: Excavation task
gym.register(
    id="srb/excavation-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.excavation.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:ExcavationCfg",
    },
)

# Example: Peg in Hole Assembly task
gym.register(
    id="srb/peg_in_hole_assembly-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.peg_in_hole_assembly.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:PegInHoleCfg",
    },
)

# Example: Sample Collection task
gym.register(
    id="srb/sample_collection-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.sample_collection.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:SampleCollectionCfg",
    },
)

# Example: Screwdriving task
gym.register(
    id="srb/screwdriving-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.screwdriving.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:ScrewdrivingCfg",
    },
)

# Example: Solar Panel Assembly task
gym.register(
    id="srb/solar_panel_assembly-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.manipulation.solar_panel_assembly.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:SolarPanelAssemblyCfg",
    },
)

# Example: Landing task
gym.register(
    id="srb/landing-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.mobile.landing.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:LandingCfg",
    },
)

# Example: Velocity Tracking task
gym.register(
    id="srb/velocity_tracking-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.mobile.velocity_tracking.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:VelocityTrackingCfg",
    },
)

# Example: Waypoint Navigation task
gym.register(
    id="srb/waypoint_navigation-PF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "srb.tasks.mobile.waypoint_navigation.task:TaskCfg",
        "pf_cfg_entry_point": "srb_agent_cfg:WaypointNavigationCfg",
    },
)