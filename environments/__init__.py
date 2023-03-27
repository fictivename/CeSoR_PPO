from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

# register(
#     'AntDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# register(
#     'AntDir2D-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200,
# )
#
# register(
#     'AntGoal-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# register(
#     'AntMass-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_mass:AntMassEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahMass-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_mass:HalfCheetahMassEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahBody-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_body:HalfCheetahBodyEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# register(
#     'HumanoidDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# register(
#     'HumanoidVel-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_vel:HumanoidVelEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# register(
#     'HumanoidMass-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_mass:HumanoidMassEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# register(
#     'HumanoidBody-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_body:HumanoidBodyEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )
#
# # - randomised dynamics
#
# register(
#     id='Walker2DRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
#     max_episode_steps=200
# )
#
# register(
#     id='HopperRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
#     max_episode_steps=200
# )
