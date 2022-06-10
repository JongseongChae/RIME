import os
import time
from collections import deque

import numpy as np
import torch

from algorithm import algo, utils
from algorithm.algo import gail
from algorithm.arguments import get_args
from algorithm.envs import make_vec_envs
from algorithm.model import Policy
from algorithm.storage import RolloutStorage
from algorithm.evaluation import evaluate

from stable_baselines3.common.running_mean_std import RunningMeanStd


def learn_the_2_sampled_envs(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_name1 = args.expert_path1.split("_")[1]
    env_name2 = args.expert_path2.split("_")[1]
    env_par1 = env_name1.split("-")[0][-4:]
    env_par2 = env_name2.split("-")[0][-4:]
    assert env_name1.split("-")[0][:2] == args.env_name[:2]
    assert env_name2.split("-")[0][:2] == args.env_name[:2]

    excute_name = "2_sampled_envs_" + str(args.algo_name) + "_" + str(env_par1) + str(env_par2) + "_" \
                  + str(args.env_name) + "_" + str(args.env_parameter) + "_" + str(args.seed) + "seed"

    if args.log_dir != None:
        log_dir = os.path.expanduser(args.log_dir) + excute_name
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir + "_env1")
        utils.cleanup_log_dir(log_dir + "_env2")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs1 = make_vec_envs(env_name1, args.seed, args.num_processes, args.gamma, log_dir + "_env1", device, False, use_rms=False)
    envs2 = make_vec_envs(env_name2, args.seed, args.num_processes, args.gamma, log_dir + "_env2", device, False, use_rms=False)
    if not args.no_rms:
        total_obs_rms = RunningMeanStd(shape=envs1.observation_space.shape)
        def total_obsfilt(obs, update=False):
            return np.clip((obs - total_obs_rms.mean) / np.sqrt(total_obs_rms.var + 1e-8), -10.0, 10.0)

    actor_critic = Policy(envs1.observation_space.shape, envs1.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                     lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)


    assert len(envs1.observation_space.shape) == 1
    assert len(envs2.observation_space.shape) == 1

    if "rime+wsd" in args.algo_name.lower():
        discr1 = gail.WeightShared_discriminator(2, envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.WeightShared_discriminator(2, envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
    elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
        discr11 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr12 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)

        discr21 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr22 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
    elif "mixture" in args.algo_name.lower():
        discr = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
    elif "single" in args.algo_name.lower():
        discr1 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
    else:
        print("Check algo-name")
        import pdb; pdb.set_trace()

    file_name1 = "./expert_demonstrations/" + args.expert_path1 + ".pt"
    file_name2 = "./expert_demonstrations/" + args.expert_path2 + ".pt"

    expert_dataset1 = gail.ExpertDataset(file_name1, num_trajectories=0, subsample_frequency=1)
    expert_dataset2 = gail.ExpertDataset(file_name2, num_trajectories=0, subsample_frequency=1)

    drop_last1 = len(expert_dataset1) > args.gail_batch_size
    drop_last2 = len(expert_dataset2) > args.gail_batch_size

    gail_train_loader1 = torch.utils.data.DataLoader(dataset=expert_dataset1, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last1)
    gail_train_loader2 = torch.utils.data.DataLoader(dataset=expert_dataset2, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last2)

    rollouts1 = RolloutStorage(args.num_steps, args.num_processes, envs1.observation_space.shape, envs1.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts2 = RolloutStorage(args.num_steps, args.num_processes, envs2.observation_space.shape, envs2.action_space, actor_critic.recurrent_hidden_state_size)

    obs1 = envs1.reset()
    obs2 = envs2.reset()
    if not args.no_rms:
        total_obs_rms.update(obs1)
        total_obs_rms.update(obs2)
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    rollouts1.obs[0].copy_(obs1)
    rollouts1.to(device)
    rollouts2.obs[0].copy_(obs2)
    rollouts2.to(device)

    episode_rewards1 = deque(maxlen=20)
    episode_rewards2 = deque(maxlen=20)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value1, action1, action_log_prob1, recurrent_hidden_states1 = actor_critic.act(rollouts1.obs[step],
                                                                                               rollouts1.recurrent_hidden_states[step],
                                                                                               rollouts1.masks[step])
                value2, action2, action_log_prob2, recurrent_hidden_states2 = actor_critic.act(rollouts2.obs[step],
                                                                                               rollouts2.recurrent_hidden_states[step],
                                                                                               rollouts2.masks[step])

            # Obser reward and next obs
            obs1, _, done1, infos1 = envs1.step(action1)
            obs2, _, done2, infos2 = envs2.step(action2)

            if not args.no_rms:
                temp_obs1 = obs1
                temp_obs2 = obs2
                obs1 = total_obsfilt(obs1)
                obs2 = total_obsfilt(obs2)
                if j < args.rms_iters + 1:
                    total_obs_rms.update(temp_obs1)
                    total_obs_rms.update(temp_obs2)

            obs1 = torch.from_numpy(obs1).float().to(device)
            obs2 = torch.from_numpy(obs2).float().to(device)

            for info1 in infos1:
                if 'episode' in info1.keys():
                    episode_rewards1.append(info1['episode']['r'])
            for info2 in infos2:
                if 'episode' in info2.keys():
                    episode_rewards2.append(info2['episode']['r'])

            # If done then clean the history of observations.
            masks1 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done1])
            masks2 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done2])
            bad_masks1 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos1])
            bad_masks2 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos2])
            rollouts1.insert(obs1, recurrent_hidden_states1, action1, action_log_prob1, value1, torch.zeros(1, 1), masks1, bad_masks1)
            rollouts2.insert(obs2, recurrent_hidden_states2, action2, action_log_prob2, value2, torch.zeros(1, 1), masks2, bad_masks2)

        with torch.no_grad():
            next_value1 = actor_critic.get_value(rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1], rollouts1.masks[-1]).detach()
            next_value2 = actor_critic.get_value(rollouts2.obs[-1], rollouts2.recurrent_hidden_states[-1], rollouts2.masks[-1]).detach()


        if j >= args.rms_iters:
            envs1.venv.eval()
            envs2.venv.eval()
        gail_epoch = args.gail_epoch
        if j < args.rms_iters:
            gail_epoch = args.initial_dstep  # Warm up
        for _ in range(gail_epoch):
            if "rime+wsd" in args.algo_name.lower():
                discr1.update_WSD_2env(rollouts1, gail_train_loader1, gail_train_loader2, rms=total_obs_rms)
                discr2.update_WSD_2env(rollouts2, gail_train_loader1, gail_train_loader2, rms=total_obs_rms)
            elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
                discr11.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr12.update_gp(gail_train_loader2, rollouts1, total_obs_rms)
                discr21.update_gp(gail_train_loader1, rollouts2, total_obs_rms)
                discr22.update_gp(gail_train_loader2, rollouts2, total_obs_rms)
            elif "mixture" in args.algo_name.lower():
                discr.update_gp_mixture_2envs(gail_train_loader1, gail_train_loader2, rollouts1, rollouts2, total_obs_rms)
            elif "single" in args.algo_name.lower():
                discr1.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr2.update_gp(gail_train_loader2, rollouts2, total_obs_rms)

        roll_temp_1 = [[], []]
        roll_temp_2 = [[], []]
        for step in range(args.num_steps):
            if "rime+wsd" in args.algo_name.lower():
                rollouts1.rewards[step] = torch.min(discr1.predict_reward(0, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                    discr1.predict_reward(1, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))

                rollouts2.rewards[step] = torch.min(discr2.predict_reward(0, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                    discr2.predict_reward(1, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
            elif ("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower()):
                rollouts1.rewards[step] = torch.min(discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                    discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))

                rollouts2.rewards[step] = torch.min(discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                    discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
            elif "omme" in args.algo_name.lower():
                roll_temp_1[0].append(discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[1].append(discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))

                roll_temp_2[0].append(discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[1].append(discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
            elif "mixture" in args.algo_name.lower():
                rollouts1.rewards[step] = discr.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])
            elif "single" in args.algo_name.lower():
                rollouts1.rewards[step] = discr1.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr2.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])

        if "omme" in args.algo_name.lower():
            index_roll1 = np.argmin([torch.mean(torch.tensor(roll_temp_1[0])), torch.mean(torch.tensor(roll_temp_1[1]))])
            index_roll2 = np.argmin([torch.mean(torch.tensor(roll_temp_2[0])), torch.mean(torch.tensor(roll_temp_2[1]))])
            for step in range(args.num_steps):
                rollouts1.rewards[step] = roll_temp_1[index_roll1][step]
                rollouts2.rewards[step] = roll_temp_2[index_roll2][step]

        rollouts1.compute_returns(next_value1, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts2.compute_returns(next_value2, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        _, _, _ = agent.update_2envs(rollouts1, rollouts2)

        rollouts1.after_update()
        rollouts2.after_update()


        if len(episode_rewards1) > 1 and len(episode_rewards2) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {} \n Last {} training episodes \n mean/median reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f} \n min/max reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f}\n"
                  .format(j, total_num_steps, len(episode_rewards1),
                          np.mean(episode_rewards1), np.median(episode_rewards1),
                          np.mean(episode_rewards2), np.median(episode_rewards2),
                          np.min(episode_rewards1), np.max(episode_rewards1),
                          np.min(episode_rewards2), np.max(episode_rewards2)))

    if (len(episode_rewards1) > 1) and (len(episode_rewards2) > 1):
        perfs = []
        env_list = []
        env_par_list_1par = ['010', '015', '020', '025', '030', '035', '040', '045', '050', '055', '060', '065', '070',
                             '075', '080', '085', '090', '095', 'nominal', '105', '110', '115', '120', '125', '130',
                             '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '185', '190', '195',
                             '200', '205', '210', '215', '220', '225', '230']

        env_n = args.env_name.split("-")
        for par in env_par_list_1par:
            if par == 'nominal':
                env_list.append(env_n[0] + '-' + env_n[1])
            else:
                if args.env_parameter == 'gravity':
                    env_list.append(env_n[0] + par + 'g-' + env_n[1])
                elif args.env_parameter == 'mass':
                    env_list.append(env_n[0] + par + 'm-' + env_n[1])

        for env_name in env_list:
            perfs.append(evaluate(actor_critic, total_obs_rms, env_name, args.seed, args.num_processes, eval_log_dir, device))

    return [perfs, excute_name]

def learn_the_3_sampled_envs(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_name1 = args.expert_path1.split("_")[1]
    env_name2 = args.expert_path2.split("_")[1]
    env_name3 = args.expert_path3.split("_")[1]
    env_par1 = env_name1.split("-")[0][-4:]
    env_par2 = env_name2.split("-")[0][-4:]
    env_par3 = env_name3.split("-")[0][-4:]
    assert env_name1.split("-")[0][:2] == args.env_name[:2]
    assert env_name2.split("-")[0][:2] == args.env_name[:2]
    assert env_name3.split("-")[0][:2] == args.env_name[:2]

    excute_name = "3_sampled_envs_" + str(args.algo_name) + "_" + str(env_par1) + str(env_par2) + str(env_par3) + "_" \
                  + str(args.env_name) + "_" + str(args.env_parameter) + "_" + str(args.seed) + "seed"

    if args.log_dir != None:
        log_dir = os.path.expanduser(args.log_dir) + excute_name
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir + "_env1")
        utils.cleanup_log_dir(log_dir + "_env2")
        utils.cleanup_log_dir(log_dir + "_env3")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs1 = make_vec_envs(env_name1, args.seed, args.num_processes, args.gamma, log_dir + "_env1", device, False, use_rms=False)
    envs2 = make_vec_envs(env_name2, args.seed, args.num_processes, args.gamma, log_dir + "_env2", device, False, use_rms=False)
    envs3 = make_vec_envs(env_name3, args.seed, args.num_processes, args.gamma, log_dir + "_env3", device, False, use_rms=False)
    if not args.no_rms:
        total_obs_rms = RunningMeanStd(shape=envs1.observation_space.shape)
        def total_obsfilt(obs, update=False):
            return np.clip((obs - total_obs_rms.mean) / np.sqrt(total_obs_rms.var + 1e-8), -10.0, 10.0)

    actor_critic = Policy(envs1.observation_space.shape,envs1.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                     lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)


    assert len(envs1.observation_space.shape) == 1
    assert len(envs2.observation_space.shape) == 1
    assert len(envs3.observation_space.shape) == 1

    if "rime+wsd" in args.algo_name.lower():
        discr1 = gail.WeightShared_discriminator(3, envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.WeightShared_discriminator(3, envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr3 = gail.WeightShared_discriminator(3, envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
    elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
        discr11 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr12 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr13 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)

        discr21 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr22 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr23 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)

        discr31 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr32 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr33 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
    elif "mixture" in args.algo_name.lower():
        discr = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
    elif "single" in args.algo_name.lower():
        discr1 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr3 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
    else:
        print("Check algo-name")
        import pdb; pdb.set_trace()

    file_name1 = "./expert_demonstrations/" + args.expert_path1 + ".pt"
    file_name2 = "./expert_demonstrations/" + args.expert_path2 + ".pt"
    file_name3 = "./expert_demonstrations/" + args.expert_path3 + ".pt"

    expert_dataset1 = gail.ExpertDataset(file_name1, num_trajectories=0, subsample_frequency=1)
    expert_dataset2 = gail.ExpertDataset(file_name2, num_trajectories=0, subsample_frequency=1)
    expert_dataset3 = gail.ExpertDataset(file_name3, num_trajectories=0, subsample_frequency=1)

    drop_last1 = len(expert_dataset1) > args.gail_batch_size
    drop_last2 = len(expert_dataset2) > args.gail_batch_size
    drop_last3 = len(expert_dataset3) > args.gail_batch_size

    gail_train_loader1 = torch.utils.data.DataLoader(dataset=expert_dataset1, batch_size=args.gail_batch_size,
                                                     shuffle=True, drop_last=drop_last1)
    gail_train_loader2 = torch.utils.data.DataLoader(dataset=expert_dataset2, batch_size=args.gail_batch_size,
                                                     shuffle=True, drop_last=drop_last2)
    gail_train_loader3 = torch.utils.data.DataLoader(dataset=expert_dataset3, batch_size=args.gail_batch_size,
                                                     shuffle=True, drop_last=drop_last3)

    rollouts1 = RolloutStorage(args.num_steps, args.num_processes, envs1.observation_space.shape, envs1.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts2 = RolloutStorage(args.num_steps, args.num_processes, envs2.observation_space.shape, envs2.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts3 = RolloutStorage(args.num_steps, args.num_processes, envs3.observation_space.shape, envs3.action_space, actor_critic.recurrent_hidden_state_size)

    obs1 = envs1.reset()
    obs2 = envs2.reset()
    obs3 = envs3.reset()
    if not args.no_rms:
        total_obs_rms.update(obs1)
        total_obs_rms.update(obs2)
        total_obs_rms.update(obs3)
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    obs3 = torch.from_numpy(obs3).float().to(device)
    rollouts1.obs[0].copy_(obs1)
    rollouts1.to(device)
    rollouts2.obs[0].copy_(obs2)
    rollouts2.to(device)
    rollouts3.obs[0].copy_(obs3)
    rollouts3.to(device)

    episode_rewards1 = deque(maxlen=20)
    episode_rewards2 = deque(maxlen=20)
    episode_rewards3 = deque(maxlen=20)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value1, action1, action_log_prob1, recurrent_hidden_states1 = actor_critic.act(rollouts1.obs[step],
                                                                                               rollouts1.recurrent_hidden_states[step],
                                                                                               rollouts1.masks[step])
                value2, action2, action_log_prob2, recurrent_hidden_states2 = actor_critic.act(rollouts2.obs[step],
                                                                                               rollouts2.recurrent_hidden_states[step],
                                                                                               rollouts2.masks[step])
                value3, action3, action_log_prob3, recurrent_hidden_states3 = actor_critic.act(rollouts3.obs[step],
                                                                                               rollouts3.recurrent_hidden_states[step],
                                                                                               rollouts3.masks[step])

            # Obser reward and next obs
            obs1, _, done1, infos1 = envs1.step(action1)
            obs2, _, done2, infos2 = envs2.step(action2)
            obs3, _, done3, infos3 = envs3.step(action3)

            if not args.no_rms:
                temp_obs1 = obs1
                temp_obs2 = obs2
                temp_obs3 = obs3
                obs1 = total_obsfilt(obs1)
                obs2 = total_obsfilt(obs2)
                obs3 = total_obsfilt(obs3)
                if j < args.rms_iters + 1:
                    total_obs_rms.update(temp_obs1)
                    total_obs_rms.update(temp_obs2)
                    total_obs_rms.update(temp_obs3)

            obs1 = torch.from_numpy(obs1).float().to(device)
            obs2 = torch.from_numpy(obs2).float().to(device)
            obs3 = torch.from_numpy(obs3).float().to(device)

            for info1 in infos1:
                if 'episode' in info1.keys():
                    episode_rewards1.append(info1['episode']['r'])
            for info2 in infos2:
                if 'episode' in info2.keys():
                    episode_rewards2.append(info2['episode']['r'])
            for info3 in infos3:
                if 'episode' in info3.keys():
                    episode_rewards3.append(info3['episode']['r'])

            # If done then clean the history of observations.
            masks1 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done1])
            masks2 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done2])
            masks3 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done3])
            bad_masks1 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos1])
            bad_masks2 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos2])
            bad_masks3 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos3])
            rollouts1.insert(obs1, recurrent_hidden_states1, action1, action_log_prob1, value1, torch.zeros(1, 1), masks1, bad_masks1)
            rollouts2.insert(obs2, recurrent_hidden_states2, action2, action_log_prob2, value2, torch.zeros(1, 1), masks2, bad_masks2)
            rollouts3.insert(obs3, recurrent_hidden_states3, action3, action_log_prob3, value3, torch.zeros(1, 1), masks3, bad_masks3)

        with torch.no_grad():
            next_value1 = actor_critic.get_value(rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1], rollouts1.masks[-1]).detach()
            next_value2 = actor_critic.get_value(rollouts2.obs[-1], rollouts2.recurrent_hidden_states[-1], rollouts2.masks[-1]).detach()
            next_value3 = actor_critic.get_value(rollouts3.obs[-1], rollouts3.recurrent_hidden_states[-1], rollouts3.masks[-1]).detach()


        if j >= args.rms_iters:
            envs1.venv.eval()
            envs2.venv.eval()
            envs3.venv.eval()
        gail_epoch = args.gail_epoch
        if j < args.rms_iters:
            gail_epoch = args.initial_dstep  # Warm up
        for _ in range(gail_epoch):
            if "rime+wsd" in args.algo_name.lower():
                discr1.update_WSD_3env(rollouts1, gail_train_loader1, gail_train_loader2, gail_train_loader3, rms=total_obs_rms)
                discr2.update_WSD_3env(rollouts2, gail_train_loader1, gail_train_loader2, gail_train_loader3, rms=total_obs_rms)
                discr3.update_WSD_3env(rollouts3, gail_train_loader1, gail_train_loader2, gail_train_loader3, rms=total_obs_rms)
            elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
                discr11.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr12.update_gp(gail_train_loader2, rollouts1, total_obs_rms)
                discr13.update_gp(gail_train_loader3, rollouts1, total_obs_rms)
                discr21.update_gp(gail_train_loader1, rollouts2, total_obs_rms)
                discr22.update_gp(gail_train_loader2, rollouts2, total_obs_rms)
                discr23.update_gp(gail_train_loader3, rollouts2, total_obs_rms)
                discr31.update_gp(gail_train_loader1, rollouts3, total_obs_rms)
                discr32.update_gp(gail_train_loader2, rollouts3, total_obs_rms)
                discr33.update_gp(gail_train_loader3, rollouts3, total_obs_rms)
            elif "mixture" in args.algo_name.lower():
                discr.update_gp_mixture_3envs(gail_train_loader1, gail_train_loader2, gail_train_loader3, rollouts1, rollouts2, rollouts3, total_obs_rms)
            elif "single" in args.algo_name.lower():
                discr1.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr2.update_gp(gail_train_loader2, rollouts2, total_obs_rms)
                discr3.update_gp(gail_train_loader3, rollouts3, total_obs_rms)

        roll_temp_1 = [[],[],[]]
        roll_temp_2 = [[],[],[]]
        roll_temp_3 = [[],[],[]]
        for step in range(args.num_steps):
            if "rime+wsd" in args.algo_name.lower():
                rollouts1.rewards[step] = torch.min(torch.cat([discr1.predict_reward(0, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr1.predict_reward(1, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr1.predict_reward(2, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])]))

                rollouts2.rewards[step] = torch.min(torch.cat([discr2.predict_reward(0, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr2.predict_reward(1, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr2.predict_reward(2, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])]))

                rollouts3.rewards[step] = torch.min(torch.cat([discr3.predict_reward(0, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr3.predict_reward(1, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr3.predict_reward(2, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])]))
            elif ("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower()):
                rollouts1.rewards[step] = torch.min(torch.cat([discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr13.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])]))

                rollouts2.rewards[step] = torch.min(torch.cat([discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr23.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])]))

                rollouts3.rewards[step] = torch.min(torch.cat([discr31.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr32.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr33.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])]))
            elif "omme" in args.algo_name.lower():
                roll_temp_1[0].append(discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[1].append(discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[2].append(discr13.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))

                roll_temp_2[0].append(discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[1].append(discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[2].append(discr23.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))

                roll_temp_3[0].append(discr31.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
                roll_temp_3[1].append(discr32.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
                roll_temp_3[2].append(discr33.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
            elif "mixture" in args.algo_name.lower():
                rollouts1.rewards[step] = discr.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])
                rollouts3.rewards[step] = discr.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])
            elif "single" in args.algo_name.lower():
                rollouts1.rewards[step] = discr1.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr2.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])
                rollouts3.rewards[step] = discr3.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])

        if "omme" in args.algo_name.lower():
            index_roll1 = np.argmin([torch.mean(torch.tensor(roll_temp_1[0])), torch.mean(torch.tensor(roll_temp_1[1])), torch.mean(torch.tensor(roll_temp_1[2]))])
            index_roll2 = np.argmin([torch.mean(torch.tensor(roll_temp_2[0])), torch.mean(torch.tensor(roll_temp_2[1])), torch.mean(torch.tensor(roll_temp_2[2]))])
            index_roll3 = np.argmin([torch.mean(torch.tensor(roll_temp_3[0])), torch.mean(torch.tensor(roll_temp_3[1])), torch.mean(torch.tensor(roll_temp_3[2]))])
            for step in range(args.num_steps):
                rollouts1.rewards[step] = roll_temp_1[index_roll1][step]
                rollouts2.rewards[step] = roll_temp_2[index_roll2][step]
                rollouts3.rewards[step] = roll_temp_3[index_roll3][step]

        rollouts1.compute_returns(next_value1, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts2.compute_returns(next_value2, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts3.compute_returns(next_value3, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        _, _, _ = agent.update_3envs(rollouts1, rollouts2, rollouts3)

        rollouts1.after_update()
        rollouts2.after_update()
        rollouts3.after_update()

        if len(episode_rewards1) > 1 and len(episode_rewards2) > 1 and len(episode_rewards3) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {} \n Last {} training episodes \n mean/median reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f}, reward3 {:.1f}/{:.1f} \n min/max reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f}, reward3 {:.1f}/{:.1f}\n"
                  .format(j, total_num_steps, len(episode_rewards1),
                          np.mean(episode_rewards1), np.median(episode_rewards1),
                          np.mean(episode_rewards2), np.median(episode_rewards2),
                          np.mean(episode_rewards3), np.median(episode_rewards3),
                          np.min(episode_rewards1), np.max(episode_rewards1),
                          np.min(episode_rewards2), np.max(episode_rewards2),
                          np.min(episode_rewards3), np.max(episode_rewards3)))

    if (len(episode_rewards1) > 1) and (len(episode_rewards2) > 1) and (len(episode_rewards3) > 1):
        perfs = []
        env_list = []
        env_par_list_1par = ['010', '015', '020', '025', '030', '035', '040', '045', '050', '055', '060', '065', '070',
                             '075', '080', '085', '090', '095', 'nominal', '105', '110', '115', '120', '125', '130',
                             '135', '140', '145', '150', '155', '160', '165', '170', '175', '180', '185', '190', '195',
                             '200', '205', '210', '215', '220', '225', '230']

        env_n = args.env_name.split("-")
        for par in env_par_list_1par:
            if par == 'nominal':
                env_list.append(env_n[0] + '-' + env_n[1])
            else:
                if args.env_parameter == 'gravity':
                    env_list.append(env_n[0] + par + 'g-' + env_n[1])
                elif args.env_parameter == 'mass':
                    env_list.append(env_n[0] + par + 'm-' + env_n[1])

        for env_name in env_list:
            perfs.append(evaluate(actor_critic, total_obs_rms, env_name, args.seed, args.num_processes, eval_log_dir, device))

    return [perfs, excute_name]

def learn_the_4_sampled_envs(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_name1 = args.expert_path1.split("_")[1]
    env_name2 = args.expert_path2.split("_")[1]
    env_name3 = args.expert_path3.split("_")[1]
    env_name4 = args.expert_path4.split("_")[1]

    env_par1 = env_name1.split("-")[0][-8:]
    env_par2 = env_name2.split("-")[0][-8:]
    env_par3 = env_name3.split("-")[0][-8:]
    env_par4 = env_name4.split("-")[0][-8:]

    assert env_name1.split("-")[0][:2] == args.env_name[:2]
    assert env_name2.split("-")[0][:2] == args.env_name[:2]
    assert env_name3.split("-")[0][:2] == args.env_name[:2]
    assert env_name4.split("-")[0][:2] == args.env_name[:2]


    excute_name = "4_sampled_envs_" + str(args.algo_name) + "_" + str(env_par1) + str(env_par2) \
                  + str(env_par3) + str(env_par4) + "_" + str(args.env_name) + "_" + str(args.seed) + "seed"

    if args.log_dir != None:
        log_dir = os.path.expanduser(args.log_dir) + excute_name
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir + "_env1")
        utils.cleanup_log_dir(log_dir + "_env2")
        utils.cleanup_log_dir(log_dir + "_env3")
        utils.cleanup_log_dir(log_dir + "_env4")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs1 = make_vec_envs(env_name1, args.seed, args.num_processes, args.gamma, log_dir + "_env1", device, False, use_rms=False)
    envs2 = make_vec_envs(env_name2, args.seed, args.num_processes, args.gamma, log_dir + "_env2", device, False, use_rms=False)
    envs3 = make_vec_envs(env_name3, args.seed, args.num_processes, args.gamma, log_dir + "_env3", device, False, use_rms=False)
    envs4 = make_vec_envs(env_name4, args.seed, args.num_processes, args.gamma, log_dir + "_env4", device, False, use_rms=False)
    if not args.no_rms:
        total_obs_rms = RunningMeanStd(shape=envs1.observation_space.shape)
        def total_obsfilt(obs, update=False):
            return np.clip((obs - total_obs_rms.mean) / np.sqrt(total_obs_rms.var + 1e-8), -10.0, 10.0)

    actor_critic = Policy(
        envs1.observation_space.shape,
        envs1.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                     lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)

    assert len(envs1.observation_space.shape) == 1
    assert len(envs2.observation_space.shape) == 1
    assert len(envs3.observation_space.shape) == 1
    assert len(envs4.observation_space.shape) == 1

    if "rime+wsd" in args.algo_name.lower():
        discr1 = gail.WeightShared_discriminator(4, envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.WeightShared_discriminator(4, envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr3 = gail.WeightShared_discriminator(4, envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr4 = gail.WeightShared_discriminator(4, envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
    elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
        discr11 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr12 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr13 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr14 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)

        discr21 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr22 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr23 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr24 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)

        discr31 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr32 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr33 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr34 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)

        discr41 = gail.Discriminator(envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
        discr42 = gail.Discriminator(envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
        discr43 = gail.Discriminator(envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
        discr44 = gail.Discriminator(envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
    elif "mixture" in args.algo_name.lower():
        discr = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
    elif "single" in args.algo_name.lower():
        discr1 = gail.Discriminator(envs1.observation_space.shape[0] + envs1.action_space.shape[0], 100, device)
        discr2 = gail.Discriminator(envs2.observation_space.shape[0] + envs2.action_space.shape[0], 100, device)
        discr3 = gail.Discriminator(envs3.observation_space.shape[0] + envs3.action_space.shape[0], 100, device)
        discr4 = gail.Discriminator(envs4.observation_space.shape[0] + envs4.action_space.shape[0], 100, device)
    else:
        print("Check algo-name")
        import pdb; pdb.set_trace()

    file_name1 = "./expert_demonstrations/" + args.expert_path1 + ".pt"
    file_name2 = "./expert_demonstrations/" + args.expert_path2 + ".pt"
    file_name3 = "./expert_demonstrations/" + args.expert_path3 + ".pt"
    file_name4 = "./expert_demonstrations/" + args.expert_path4 + ".pt"

    expert_dataset1 = gail.ExpertDataset(file_name1, num_trajectories=0, subsample_frequency=1)
    expert_dataset2 = gail.ExpertDataset(file_name2, num_trajectories=0, subsample_frequency=1)
    expert_dataset3 = gail.ExpertDataset(file_name3, num_trajectories=0, subsample_frequency=1)
    expert_dataset4 = gail.ExpertDataset(file_name4, num_trajectories=0, subsample_frequency=1)

    drop_last1 = len(expert_dataset1) > args.gail_batch_size
    drop_last2 = len(expert_dataset2) > args.gail_batch_size
    drop_last3 = len(expert_dataset3) > args.gail_batch_size
    drop_last4 = len(expert_dataset4) > args.gail_batch_size

    gail_train_loader1 = torch.utils.data.DataLoader(dataset=expert_dataset1, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last1)
    gail_train_loader2 = torch.utils.data.DataLoader(dataset=expert_dataset2, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last2)
    gail_train_loader3 = torch.utils.data.DataLoader(dataset=expert_dataset3, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last3)
    gail_train_loader4 = torch.utils.data.DataLoader(dataset=expert_dataset4, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last4)

    rollouts1 = RolloutStorage(args.num_steps, args.num_processes, envs1.observation_space.shape, envs1.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts2 = RolloutStorage(args.num_steps, args.num_processes, envs2.observation_space.shape, envs2.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts3 = RolloutStorage(args.num_steps, args.num_processes, envs3.observation_space.shape, envs3.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts4 = RolloutStorage(args.num_steps, args.num_processes, envs4.observation_space.shape, envs4.action_space, actor_critic.recurrent_hidden_state_size)

    obs1 = envs1.reset()
    obs2 = envs2.reset()
    obs3 = envs3.reset()
    obs4 = envs4.reset()
    if not args.no_rms:
        total_obs_rms.update(obs1)
        total_obs_rms.update(obs2)
        total_obs_rms.update(obs3)
        total_obs_rms.update(obs4)
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    obs3 = torch.from_numpy(obs3).float().to(device)
    obs4 = torch.from_numpy(obs4).float().to(device)
    rollouts1.obs[0].copy_(obs1)
    rollouts1.to(device)
    rollouts2.obs[0].copy_(obs2)
    rollouts2.to(device)
    rollouts3.obs[0].copy_(obs3)
    rollouts3.to(device)
    rollouts4.obs[0].copy_(obs4)
    rollouts4.to(device)

    episode_rewards1 = deque(maxlen=20)
    episode_rewards2 = deque(maxlen=20)
    episode_rewards3 = deque(maxlen=20)
    episode_rewards4 = deque(maxlen=20)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value1, action1, action_log_prob1, recurrent_hidden_states1 = actor_critic.act(rollouts1.obs[step],
                                                                                               rollouts1.recurrent_hidden_states[step],
                                                                                               rollouts1.masks[step])
                value2, action2, action_log_prob2, recurrent_hidden_states2 = actor_critic.act(rollouts2.obs[step],
                                                                                               rollouts2.recurrent_hidden_states[step],
                                                                                               rollouts2.masks[step])
                value3, action3, action_log_prob3, recurrent_hidden_states3 = actor_critic.act(rollouts3.obs[step],
                                                                                               rollouts3.recurrent_hidden_states[step],
                                                                                               rollouts3.masks[step])
                value4, action4, action_log_prob4, recurrent_hidden_states4 = actor_critic.act(rollouts4.obs[step],
                                                                                               rollouts4.recurrent_hidden_states[step],
                                                                                               rollouts4.masks[step])

            # Obser reward and next obs
            obs1, _, done1, infos1 = envs1.step(action1)
            obs2, _, done2, infos2 = envs2.step(action2)
            obs3, _, done3, infos3 = envs3.step(action3)
            obs4, _, done4, infos4 = envs4.step(action4)

            if not args.no_rms:
                temp_obs1 = obs1
                temp_obs2 = obs2
                temp_obs3 = obs3
                temp_obs4 = obs4
                obs1 = total_obsfilt(obs1)
                obs2 = total_obsfilt(obs2)
                obs3 = total_obsfilt(obs3)
                obs4 = total_obsfilt(obs4)
                if j < args.rms_iters + 1:
                    total_obs_rms.update(temp_obs1)
                    total_obs_rms.update(temp_obs2)
                    total_obs_rms.update(temp_obs3)
                    total_obs_rms.update(temp_obs4)

            obs1 = torch.from_numpy(obs1).float().to(device)
            obs2 = torch.from_numpy(obs2).float().to(device)
            obs3 = torch.from_numpy(obs3).float().to(device)
            obs4 = torch.from_numpy(obs4).float().to(device)

            for info1 in infos1:
                if 'episode' in info1.keys():
                    episode_rewards1.append(info1['episode']['r'])
            for info2 in infos2:
                if 'episode' in info2.keys():
                    episode_rewards2.append(info2['episode']['r'])
            for info3 in infos3:
                if 'episode' in info3.keys():
                    episode_rewards3.append(info3['episode']['r'])
            for info4 in infos4:
                if 'episode' in info4.keys():
                    episode_rewards4.append(info4['episode']['r'])

            # If done then clean the history of observations.
            masks1 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done1])
            masks2 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done2])
            masks3 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done3])
            masks4 = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done4])
            bad_masks1 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos1])
            bad_masks2 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos2])
            bad_masks3 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos3])
            bad_masks4 = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos4])
            rollouts1.insert(obs1, recurrent_hidden_states1, action1, action_log_prob1, value1, torch.zeros(1, 1), masks1, bad_masks1)
            rollouts2.insert(obs2, recurrent_hidden_states2, action2, action_log_prob2, value2, torch.zeros(1, 1), masks2, bad_masks2)
            rollouts3.insert(obs3, recurrent_hidden_states3, action3, action_log_prob3, value3, torch.zeros(1, 1), masks3, bad_masks3)
            rollouts4.insert(obs4, recurrent_hidden_states4, action4, action_log_prob4, value4, torch.zeros(1, 1), masks4, bad_masks4)

        with torch.no_grad():
            next_value1 = actor_critic.get_value(rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1], rollouts1.masks[-1]).detach()
            next_value2 = actor_critic.get_value(rollouts2.obs[-1], rollouts2.recurrent_hidden_states[-1], rollouts2.masks[-1]).detach()
            next_value3 = actor_critic.get_value(rollouts3.obs[-1], rollouts3.recurrent_hidden_states[-1], rollouts3.masks[-1]).detach()
            next_value4 = actor_critic.get_value(rollouts4.obs[-1], rollouts4.recurrent_hidden_states[-1], rollouts4.masks[-1]).detach()


        if j >= args.rms_iters:
            envs1.venv.eval()
            envs2.venv.eval()
            envs3.venv.eval()
            envs4.venv.eval()
        gail_epoch = args.gail_epoch
        if j < args.rms_iters:
            gail_epoch = args.initial_dstep  # Warm up
        for _ in range(gail_epoch):
            if "rime+wsd" in args.algo_name.lower():
                discr1.update_WSD_4env(rollouts1, gail_train_loader1, gail_train_loader2, gail_train_loader3, gail_train_loader4, rms=total_obs_rms)
                discr2.update_WSD_4env(rollouts2, gail_train_loader1, gail_train_loader2, gail_train_loader3, gail_train_loader4, rms=total_obs_rms)
                discr3.update_WSD_4env(rollouts3, gail_train_loader1, gail_train_loader2, gail_train_loader3, gail_train_loader4, rms=total_obs_rms)
                discr4.update_WSD_4env(rollouts4, gail_train_loader1, gail_train_loader2, gail_train_loader3, gail_train_loader4, rms=total_obs_rms)
            elif (("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower())) or "omme" in args.algo_name.lower():
                discr11.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr12.update_gp(gail_train_loader2, rollouts1, total_obs_rms)
                discr13.update_gp(gail_train_loader3, rollouts1, total_obs_rms)
                discr14.update_gp(gail_train_loader4, rollouts1, total_obs_rms)

                discr21.update_gp(gail_train_loader1, rollouts2, total_obs_rms)
                discr22.update_gp(gail_train_loader2, rollouts2, total_obs_rms)
                discr23.update_gp(gail_train_loader3, rollouts2, total_obs_rms)
                discr24.update_gp(gail_train_loader4, rollouts2, total_obs_rms)

                discr31.update_gp(gail_train_loader1, rollouts3, total_obs_rms)
                discr32.update_gp(gail_train_loader2, rollouts3, total_obs_rms)
                discr33.update_gp(gail_train_loader3, rollouts3, total_obs_rms)
                discr34.update_gp(gail_train_loader4, rollouts3, total_obs_rms)

                discr41.update_gp(gail_train_loader1, rollouts4, total_obs_rms)
                discr42.update_gp(gail_train_loader2, rollouts4, total_obs_rms)
                discr43.update_gp(gail_train_loader3, rollouts4, total_obs_rms)
                discr44.update_gp(gail_train_loader4, rollouts4, total_obs_rms)
            elif "mixture" in args.algo_name.lower():
                discr.update_gp_mixture_4envs(gail_train_loader1, gail_train_loader2, gail_train_loader3, gail_train_loader4, rollouts1, rollouts2, rollouts3, rollouts4, total_obs_rms)
            elif "single" in args.algo_name.lower():
                discr1.update_gp(gail_train_loader1, rollouts1, total_obs_rms)
                discr2.update_gp(gail_train_loader2, rollouts2, total_obs_rms)
                discr3.update_gp(gail_train_loader3, rollouts3, total_obs_rms)
                discr4.update_gp(gail_train_loader4, rollouts4, total_obs_rms)


        roll_temp_1 = [[],[],[],[]]
        roll_temp_2 = [[],[],[],[]]
        roll_temp_3 = [[],[],[],[]]
        roll_temp_4 = [[],[],[],[]]
        for step in range(args.num_steps):
            if "rime+wsd" in args.algo_name.lower():
                rollouts1.rewards[step] = torch.min(torch.cat([discr1.predict_reward(0, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr1.predict_reward(1, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr1.predict_reward(2, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr1.predict_reward(3, rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])]))

                rollouts2.rewards[step] = torch.min(torch.cat([discr2.predict_reward(0, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr2.predict_reward(1, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr2.predict_reward(2, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr2.predict_reward(3, rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])]))

                rollouts3.rewards[step] = torch.min(torch.cat([discr3.predict_reward(0, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr3.predict_reward(1, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr3.predict_reward(2, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr3.predict_reward(3, rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])]))

                rollouts4.rewards[step] = torch.min(torch.cat([discr4.predict_reward(0, rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr4.predict_reward(1, rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr4.predict_reward(2, rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr4.predict_reward(3, rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step])]))
            elif ("rime" in args.algo_name.lower()) and ("wsd" not in args.algo_name.lower()):
                rollouts1.rewards[step] = torch.min(torch.cat([discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr13.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]),
                                                               discr14.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])]))

                rollouts2.rewards[step] = torch.min(torch.cat([discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr23.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]),
                                                               discr24.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])]))

                rollouts3.rewards[step] = torch.min(torch.cat([discr31.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr32.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr33.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]),
                                                               discr34.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])]))

                rollouts4.rewards[step] = torch.min(torch.cat([discr41.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr42.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr43.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]),
                                                               discr44.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step])]))
            elif "omme" in args.algo_name.lower():
                roll_temp_1[0].append(discr11.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[1].append(discr12.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[2].append(discr13.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))
                roll_temp_1[3].append(discr14.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step]))

                roll_temp_2[0].append(discr21.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[1].append(discr22.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[2].append(discr23.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))
                roll_temp_2[3].append(discr24.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step]))

                roll_temp_3[0].append(discr31.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
                roll_temp_3[1].append(discr32.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
                roll_temp_3[2].append(discr33.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))
                roll_temp_3[3].append(discr34.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step]))

                roll_temp_4[0].append(discr41.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]))
                roll_temp_4[1].append(discr42.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]))
                roll_temp_4[2].append(discr43.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]))
                roll_temp_4[3].append(discr44.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step]))
            elif "mixture" in args.algo_name.lower():
                rollouts1.rewards[step] = discr.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])
                rollouts3.rewards[step] = discr.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])
                rollouts4.rewards[step] = discr.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step])
            elif "single" in args.algo_name.lower():
                rollouts1.rewards[step] = discr1.predict_reward(rollouts1.obs[step], rollouts1.actions[step], args.gamma, rollouts1.masks[step])
                rollouts2.rewards[step] = discr2.predict_reward(rollouts2.obs[step], rollouts2.actions[step], args.gamma, rollouts2.masks[step])
                rollouts3.rewards[step] = discr3.predict_reward(rollouts3.obs[step], rollouts3.actions[step], args.gamma, rollouts3.masks[step])
                rollouts4.rewards[step] = discr4.predict_reward(rollouts4.obs[step], rollouts4.actions[step], args.gamma, rollouts4.masks[step])

        if "omme" in args.algo_name.lower():
            index_roll1 = np.argmin([torch.mean(torch.tensor(roll_temp_1[0])), torch.mean(torch.tensor(roll_temp_1[1])),
                                     torch.mean(torch.tensor(roll_temp_1[2])), torch.mean(torch.tensor(roll_temp_1[3]))])
            index_roll2 = np.argmin([torch.mean(torch.tensor(roll_temp_2[0])), torch.mean(torch.tensor(roll_temp_2[1])),
                                     torch.mean(torch.tensor(roll_temp_2[2])), torch.mean(torch.tensor(roll_temp_2[3]))])
            index_roll3 = np.argmin([torch.mean(torch.tensor(roll_temp_3[0])), torch.mean(torch.tensor(roll_temp_3[1])),
                                     torch.mean(torch.tensor(roll_temp_3[2])), torch.mean(torch.tensor(roll_temp_3[3]))])
            index_roll4 = np.argmin([torch.mean(torch.tensor(roll_temp_4[0])), torch.mean(torch.tensor(roll_temp_4[1])),
                                     torch.mean(torch.tensor(roll_temp_4[2])), torch.mean(torch.tensor(roll_temp_4[3]))])
            for step in range(args.num_steps):
                rollouts1.rewards[step] = roll_temp_1[index_roll1][step]
                rollouts2.rewards[step] = roll_temp_2[index_roll2][step]
                rollouts3.rewards[step] = roll_temp_3[index_roll3][step]
                rollouts4.rewards[step] = roll_temp_4[index_roll4][step]

        rollouts1.compute_returns(next_value1, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts2.compute_returns(next_value2, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts3.compute_returns(next_value3, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        rollouts4.compute_returns(next_value4, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        _, _, _ = agent.update_4envs(rollouts1, rollouts2, rollouts3, rollouts4)

        rollouts1.after_update()
        rollouts2.after_update()
        rollouts3.after_update()
        rollouts4.after_update()

        if len(episode_rewards1) > 1 and len(episode_rewards2) > 1  and len(episode_rewards3) > 1 and len(episode_rewards4) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {} \n Last {} training episodes \n mean/median reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f}, reward3 {:.1f}/{:.1f}, reward4 {:.1f}/{:.1f} \n min/max reward1 {:.1f}/{:.1f}, reward2 {:.1f}/{:.1f}, reward3 {:.1f}/{:.1f}, reward4 {:.1f}/{:.1f}\n"
                  .format(j, total_num_steps, len(episode_rewards1),
                          np.mean(episode_rewards1), np.median(episode_rewards1),
                          np.mean(episode_rewards2), np.median(episode_rewards2),
                          np.mean(episode_rewards3), np.median(episode_rewards3),
                          np.mean(episode_rewards4), np.median(episode_rewards4),
                          np.min(episode_rewards1), np.max(episode_rewards1),
                          np.min(episode_rewards2), np.max(episode_rewards2),
                          np.min(episode_rewards3), np.max(episode_rewards3),
                          np.min(episode_rewards4), np.max(episode_rewards4)))

    if (len(episode_rewards1) > 1) and (len(episode_rewards2) > 1) and (len(episode_rewards3) > 1) and (len(episode_rewards4) > 1):
        perfs_2par_gm = []

        env_par_list_2par = ['050g150m', '070g150m', '090g150m', '110g150m', '130g150m', '150g150m',
                             '050g130m', '070g130m', '090g130m', '110g130m', '130g130m', '150g130m',
                             '050g110m', '070g110m', '090g110m', '110g110m', '130g110m', '150g110m',
                             '050g090m', '070g090m', '090g090m', '110g090m', '130g090m', '150g090m',
                             '050g070m', '070g070m', '090g070m', '110g070m', '130g070m', '150g070m',
                             '050g050m', '070g050m', '090g050m', '110g050m', '130g050m', '150g050m']
        env_list_gm = []
        env_n = args.env_name.split("-")
        for par in env_par_list_2par:
            env_list_gm.append(env_n[0] + par + '-' + env_n[1])

        for env_name in env_list_gm:
            perfs_2par_gm.append(evaluate(actor_critic, total_obs_rms, env_name, args.seed, args.num_processes, eval_log_dir, device))

    return [perfs_2par_gm, excute_name]


if __name__ == "__main__":
    args = get_args()
    if args.sampled_envs == 2:
        learn_the_2_sampled_envs(args)
    elif args.sampled_envs == 3:
        learn_the_3_sampled_envs(args)
    elif args.sampled_envs == 4:
        learn_the_4_sampled_envs(args)