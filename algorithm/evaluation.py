import numpy as np
import torch

from algorithm import utils
from algorithm.envs import make_vec_envs

def evaluate(actor_critic, total_obs_rms, env_name, seed, num_processes, eval_log_dir, device):
    utils.cleanup_log_dir(eval_log_dir+"/"+env_name)
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, None, eval_log_dir+"/"+env_name, device,
                              True, use_rms=False)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    obs = np.clip((obs - total_obs_rms.mean) / np.sqrt(total_obs_rms.var + 1e-8), -10.0, 10.0)
    obs = torch.from_numpy(obs).float().to(device)
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(obs, eval_recurrent_hidden_states,
                                                                          eval_masks, deterministic=True)

        obs, _, done, infos = eval_envs.step(action)
        obs = np.clip((obs - total_obs_rms.mean) / np.sqrt(total_obs_rms.var + 1e-8), -10.0, 10.0)
        obs = torch.from_numpy(obs).float().to(device)

        eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32, device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return eval_episode_rewards