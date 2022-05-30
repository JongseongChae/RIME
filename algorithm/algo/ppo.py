import torch
import torch.nn as nn
import torch.optim as optim

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_2envs(self, rollouts1, rollouts2):
        advantages1 = rollouts1.returns[:-1] - rollouts1.value_preds[:-1]
        advantages2 = rollouts2.returns[:-1] - rollouts2.value_preds[:-1]
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-5)
        advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator1 = rollouts1.recurrent_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.recurrent_generator(advantages2, self.num_mini_batch)
            else:
                data_generator1 = rollouts1.feed_forward_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.feed_forward_generator(advantages2, self.num_mini_batch)

            for sample1, sample2 in zip(data_generator1, data_generator2):
                obs_batch1, recurrent_hidden_states_batch1, actions_batch1, value_preds_batch1, return_batch1,\
                masks_batch1, old_action_log_probs_batch1, adv_targ1 = sample1
                obs_batch2, recurrent_hidden_states_batch2, actions_batch2, value_preds_batch2, return_batch2,\
                masks_batch2, old_action_log_probs_batch2, adv_targ2 = sample2

                obs_batch = torch.cat([obs_batch1, obs_batch2], dim=0)
                recurrent_hidden_states_batch = torch.cat([recurrent_hidden_states_batch1, recurrent_hidden_states_batch2], dim=0)
                actions_batch = torch.cat([actions_batch1, actions_batch2], dim=0)
                value_preds_batch = torch.cat([value_preds_batch1, value_preds_batch2], dim=0)
                return_batch = torch.cat([return_batch1, return_batch2], dim=0)
                masks_batch = torch.cat([masks_batch1, masks_batch2], dim=0)
                old_action_log_probs_batch = torch.cat([old_action_log_probs_batch1, old_action_log_probs_batch2], dim=0)
                adv_targ = torch.cat([adv_targ1, adv_targ2], dim=0)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                                                               masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_3envs(self, rollouts1, rollouts2, rollouts3):
        advantages1 = rollouts1.returns[:-1] - rollouts1.value_preds[:-1]
        advantages2 = rollouts2.returns[:-1] - rollouts2.value_preds[:-1]
        advantages3 = rollouts3.returns[:-1] - rollouts3.value_preds[:-1]
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-5)
        advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-5)
        advantages3 = (advantages3 - advantages3.mean()) / (advantages3.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator1 = rollouts1.recurrent_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.recurrent_generator(advantages2, self.num_mini_batch)
                data_generator3 = rollouts3.recurrent_generator(advantages3, self.num_mini_batch)
            else:
                data_generator1 = rollouts1.feed_forward_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.feed_forward_generator(advantages2, self.num_mini_batch)
                data_generator3 = rollouts3.feed_forward_generator(advantages3, self.num_mini_batch)

            for sample1, sample2, sample3 in zip(data_generator1, data_generator2, data_generator3):
                if len(sample1) == 8:
                    obs_batch1, recurrent_hidden_states_batch1, actions_batch1, value_preds_batch1, return_batch1,\
                    masks_batch1, old_action_log_probs_batch1, adv_targ1 = sample1
                    obs_batch2, recurrent_hidden_states_batch2, actions_batch2, value_preds_batch2, return_batch2,\
                    masks_batch2, old_action_log_probs_batch2, adv_targ2 = sample2
                    obs_batch3, recurrent_hidden_states_batch3, actions_batch3, value_preds_batch3, return_batch3,\
                    masks_batch3, old_action_log_probs_batch3, adv_targ3 = sample3
                elif len(sample1) == 9:
                    obs_batch1, recurrent_hidden_states_batch1, actions_batch1, value_preds_batch1, return_batch1, \
                    masks_batch1, old_action_log_probs_batch1, adv_targ1, _ = sample1
                    obs_batch2, recurrent_hidden_states_batch2, actions_batch2, value_preds_batch2, return_batch2, \
                    masks_batch2, old_action_log_probs_batch2, adv_targ2, _ = sample2
                    obs_batch3, recurrent_hidden_states_batch3, actions_batch3, value_preds_batch3, return_batch3, \
                    masks_batch3, old_action_log_probs_batch3, adv_targ3, _ = sample3

                obs_batch = torch.cat([obs_batch1, obs_batch2, obs_batch3], dim=0)
                recurrent_hidden_states_batch = torch.cat([recurrent_hidden_states_batch1, recurrent_hidden_states_batch2,
                                                           recurrent_hidden_states_batch3], dim=0)
                actions_batch = torch.cat([actions_batch1, actions_batch2, actions_batch3], dim=0)
                value_preds_batch = torch.cat([value_preds_batch1, value_preds_batch2, value_preds_batch3], dim=0)
                return_batch = torch.cat([return_batch1, return_batch2, return_batch3], dim=0)
                masks_batch = torch.cat([masks_batch1, masks_batch2, masks_batch3], dim=0)
                old_action_log_probs_batch = torch.cat([old_action_log_probs_batch1, old_action_log_probs_batch2,
                                                        old_action_log_probs_batch3], dim=0)
                adv_targ = torch.cat([adv_targ1, adv_targ2, adv_targ3], dim=0)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                                                               masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_4envs(self, rollouts1, rollouts2, rollouts3, rollouts4):
        advantages1 = rollouts1.returns[:-1] - rollouts1.value_preds[:-1]
        advantages2 = rollouts2.returns[:-1] - rollouts2.value_preds[:-1]
        advantages3 = rollouts3.returns[:-1] - rollouts3.value_preds[:-1]
        advantages4 = rollouts4.returns[:-1] - rollouts4.value_preds[:-1]
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-5)
        advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-5)
        advantages3 = (advantages3 - advantages3.mean()) / (advantages3.std() + 1e-5)
        advantages4 = (advantages4 - advantages4.mean()) / (advantages4.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator1 = rollouts1.recurrent_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.recurrent_generator(advantages2, self.num_mini_batch)
                data_generator3 = rollouts3.recurrent_generator(advantages3, self.num_mini_batch)
                data_generator4 = rollouts4.recurrent_generator(advantages4, self.num_mini_batch)
            else:
                data_generator1 = rollouts1.feed_forward_generator(advantages1, self.num_mini_batch)
                data_generator2 = rollouts2.feed_forward_generator(advantages2, self.num_mini_batch)
                data_generator3 = rollouts3.feed_forward_generator(advantages3, self.num_mini_batch)
                data_generator4 = rollouts4.feed_forward_generator(advantages4, self.num_mini_batch)

            for sample1, sample2, sample3, sample4 in zip(data_generator1, data_generator2, data_generator3, data_generator4):
                if len(sample1) == 8:
                    obs_batch1, recurrent_hidden_states_batch1, actions_batch1, value_preds_batch1, return_batch1,\
                    masks_batch1, old_action_log_probs_batch1, adv_targ1 = sample1
                    obs_batch2, recurrent_hidden_states_batch2, actions_batch2, value_preds_batch2, return_batch2,\
                    masks_batch2, old_action_log_probs_batch2, adv_targ2 = sample2
                    obs_batch3, recurrent_hidden_states_batch3, actions_batch3, value_preds_batch3, return_batch3,\
                    masks_batch3, old_action_log_probs_batch3, adv_targ3 = sample3
                    obs_batch4, recurrent_hidden_states_batch4, actions_batch4, value_preds_batch4, return_batch4, \
                    masks_batch4, old_action_log_probs_batch4, adv_targ4 = sample4
                elif len(sample1) == 9:
                    obs_batch1, recurrent_hidden_states_batch1, actions_batch1, value_preds_batch1, return_batch1, \
                    masks_batch1, old_action_log_probs_batch1, adv_targ1, _ = sample1
                    obs_batch2, recurrent_hidden_states_batch2, actions_batch2, value_preds_batch2, return_batch2, \
                    masks_batch2, old_action_log_probs_batch2, adv_targ2, _ = sample2
                    obs_batch3, recurrent_hidden_states_batch3, actions_batch3, value_preds_batch3, return_batch3, \
                    masks_batch3, old_action_log_probs_batch3, adv_targ3, _ = sample3
                    obs_batch4, recurrent_hidden_states_batch4, actions_batch4, value_preds_batch4, return_batch4, \
                    masks_batch4, old_action_log_probs_batch4, adv_targ4, _ = sample4

                obs_batch = torch.cat([obs_batch1, obs_batch2, obs_batch3, obs_batch4], dim=0)
                recurrent_hidden_states_batch = torch.cat([recurrent_hidden_states_batch1, recurrent_hidden_states_batch2,
                                                           recurrent_hidden_states_batch3, recurrent_hidden_states_batch4], dim=0)
                actions_batch = torch.cat([actions_batch1, actions_batch2, actions_batch3, actions_batch4], dim=0)
                value_preds_batch = torch.cat([value_preds_batch1, value_preds_batch2, value_preds_batch3, value_preds_batch4], dim=0)
                return_batch = torch.cat([return_batch1, return_batch2, return_batch3, return_batch4], dim=0)
                masks_batch = torch.cat([masks_batch1, masks_batch2, masks_batch3, masks_batch4], dim=0)
                old_action_log_probs_batch = torch.cat([old_action_log_probs_batch1, old_action_log_probs_batch2,
                                                        old_action_log_probs_batch3, old_action_log_probs_batch4], dim=0)
                adv_targ = torch.cat([adv_targ1, adv_targ2, adv_targ3, adv_targ4], dim=0)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                                                               masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
