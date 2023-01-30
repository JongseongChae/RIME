from gym.envs.registration import registry, register, make, spec

# ----------------------- custom_env -------------------------------
env_name = ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant']
env_par_list_1par = ['010', '015', '020', '025', '030', '035', '040', '045', '050', '055', '060', '065', '070',
                             '075', '080', '085', '090', '095', 'nominal', '105', '110', '115', '120', '125', '130',
                             '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '185', '190', '195',
                             '200', '205', '210', '215', '220', '225', '230']
env_par_list_2par = ['050g150m', '070g150m', '090g150m', '110g150m', '130g150m', '150g150m',
                             '050g130m', '070g130m', '090g130m', '110g130m', '130g130m', '150g130m',
                             '050g110m', '070g110m', '090g110m', '110g110m', '130g110m', '150g110m',
                             '050g090m', '070g090m', '090g090m', '110g090m', '130g090m', '150g090m',
                             '050g070m', '070g070m', '090g070m', '110g070m', '130g070m', '150g070m',
                             '050g050m', '070g050m', '090g050m', '110g050m', '130g050m', '150g050m']

for env_par in env_par_list_2par:
    # -------------------- Hopper_with_2parameters(gravity&mass) ----------------------
    register(
        id=env_name[0] + env_par + '-v2',
        entry_point='environments.mujoco.hopper_with2par:' + env_name[0] + 'Env_2par' + env_par,
        max_episode_steps=1000,
        reward_threshold=3800.0,
    )
    # -------------------- Walker2d_with_2parameters(gravity&mass) ----------------------
    register(
        id=env_name[1] + env_par + '-v2',
        max_episode_steps=1000,
        entry_point='environments.mujoco.walker2d_with2par:' + env_name[1] + 'Env_2par' + env_par,
    )
    # -------------------- HalfCheetah_with_2parameters(gravity&mass) ----------------------
    register(
        id=env_name[2] + env_par + '-v2',
        entry_point='environments.mujoco.half_cheetah_with2par:' + env_name[2] + 'Env_2par' + env_par,
        max_episode_steps=1000,
        reward_threshold=4800.0,
    )
    # -------------------- Ant_with_2parameters(gravity&mass) ----------------------
    register(
        id=env_name[3] + env_par + '-v2',
        entry_point='environments.mujoco.ant_with2par:' + env_name[3] + 'Env_2par' + env_par,
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )

for env_par in env_par_list_1par:
    # --------------------Hopper----------------------
    register(
        id=env_name[0] + env_par + 'g-v2',
        entry_point='environments.mujoco.hopper_gravity:' + env_name[0] + 'Env_' + env_par + "g",
        max_episode_steps=1000,
        reward_threshold=3800.0,
    )
    register(
        id=env_name[0] + env_par + 'm-v2',
        entry_point='environments.mujoco.hopper_mass:' + env_name[0] + 'Env_' + env_par + "m",
        max_episode_steps=1000,
        reward_threshold=3800.0,
    )
    # --------------------Walker2d----------------------
    register(
        id=env_name[1] + env_par + 'g-v2',
        max_episode_steps=1000,
        entry_point='environments.mujoco.walker2d_gravity:' + env_name[1] + 'Env_' + env_par + "g",
    )
    register(
        id=env_name[1] + env_par + 'm-v2',
        max_episode_steps=1000,
        entry_point='environments.mujoco.walker2d_mass:' + env_name[1] + 'Env_' + env_par + "m",
    )
    # --------------------HalfCheetah----------------------
    register(
        id=env_name[2] + env_par + 'g-v2',
        entry_point='environments.mujoco.half_cheetah_gravity:' + env_name[2] + 'Env_' + env_par + "g",
        max_episode_steps=1000,
        reward_threshold=4800.0,
    )
    register(
        id=env_name[2] + env_par + 'm-v2',
        entry_point='environments.mujoco.half_cheetah_mass:' + env_name[2] + 'Env_' + env_par + "m",
        max_episode_steps=1000,
        reward_threshold=4800.0,
    )
    # --------------------Ant----------------------
    register(
        id=env_name[3] + env_par + 'g-v2',
        entry_point='environments.mujoco.ant_gravity:' + env_name[3] + 'Env_' + env_par + "g",
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )
    register(
        id=env_name[3] + env_par + 'm-v2',
        entry_point='environments.mujoco.ant_mass:' + env_name[3] + 'Env_' + env_par + "m",
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )

# custom_envs ------------------------------------------------------------------------------
