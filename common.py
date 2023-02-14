import or_gym


def make_env(env_name, env_seed=None):
    if env_name == 'InvManagement-v1':
        env = or_gym.make('InvManagement-v1', env_seed=env_seed)

    if env_name == 'NetworkManagement-v1':
        env = or_gym.make('NetworkManagement-v1', env_seed=env_seed)

    if env_name == 'NetworkManagement-v1-100':
        env = or_gym.make('NetworkManagement-v1', env_seed=env_seed, num_periods=100)

    if env_name == 'NetworkManagement-v1-100-alt':
        env = or_gym.make('NetworkManagement-v1', env_seed=env_seed, num_periods=100)

    return env


make_env('NetworkManagement-v1-100-alt')