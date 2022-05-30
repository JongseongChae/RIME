import os

import gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)

        return env
    return _thunk

# [!!] We use env without the RMS (Running Mean Std.)
# [!!] because we already use the RMS for multiple sampled environments in train_n_evaluate.py.
def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, allow_early_resets, use_rms=False):
    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if (gamma is None) and (not use_rms):
            envs = VecNormalize_no_rms(envs, training=False, norm_obs=False, norm_reward=False)
        elif (gamma is not None) and (not use_rms):
            envs = VecNormalize_no_rms(envs, gamma=gamma, training=False, norm_obs=False)
        elif use_rms:
            print("Check for using rms")

    if (not use_rms):
        envs = VecPyTorch_no_rms(envs, device)
    else:
        print("Check for using rms")

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch_no_rms(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch_no_rms, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize_no_rms(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize_no_rms, self).__init__(*args, **kwargs)
        self.training = False

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                import pdb;pdb.set_trace()
                print("now")
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

