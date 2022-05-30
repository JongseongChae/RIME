import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from stable_baselines3.common.running_mean_std import RunningMeanStd
from algorithm.storage import discr_update_generator

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())


    def compute_grad_pen(self, expert_state, expert_action, policy_state, policy_action, lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_gp(self, expert_loader, rollouts, rms=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))
            expert_state, expert_action = expert_batch
            expert_state = np.clip((expert_state.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action, policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n



    def compute_grad_pen_mixture_2envs(self, expert_state1, expert_action1, expert_state2, expert_action2,
                                       policy_state1, policy_action1, policy_state2, policy_action2, lambda_=10):
        expert_state = torch.cat([expert_state1, expert_state2], dim=0)
        expert_action = torch.cat([expert_action1, expert_action2], dim=0)
        policy_state = torch.cat([policy_state1, policy_state2], dim=0)
        policy_action = torch.cat([policy_action1, policy_action2], dim=0)

        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_gp_mixture_2envs(self, expert_loader1, expert_loader2, rollouts1, rollouts2, rms=None):
        self.train()

        policy_data_generator1 = rollouts1.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)
        policy_data_generator2 = rollouts2.feed_forward_generator(None, mini_batch_size=expert_loader2.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, policy_batch1, policy_batch2 \
                in zip(expert_loader1, expert_loader2, policy_data_generator1, policy_data_generator2):
            policy_state1, policy_action1 = policy_batch1[0], policy_batch1[2]
            policy_state2, policy_action2 = policy_batch2[0], policy_batch2[2]
            policy_d1 = self.trunk(torch.cat([policy_state1, policy_action1], dim=1))
            policy_d2 = self.trunk(torch.cat([policy_state2, policy_action2], dim=1))
            expert_state1, expert_action1 = expert_batch1
            expert_state2, expert_action2 = expert_batch2
            expert_state1 = np.clip((expert_state1.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state1 = torch.FloatTensor(expert_state1).to(self.device)
            expert_action1 = expert_action1.to(self.device)
            expert_state2 = np.clip((expert_state2.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state2 = torch.FloatTensor(expert_state2).to(self.device)
            expert_action2 = expert_action2.to(self.device)
            expert_d1 = self.trunk(torch.cat([expert_state1, expert_action1], dim=1))
            expert_d2 = self.trunk(torch.cat([expert_state2, expert_action2], dim=1))

            expert_loss1 = F.binary_cross_entropy_with_logits(expert_d1, torch.ones(expert_d1.size()).to(self.device))
            expert_loss2 = F.binary_cross_entropy_with_logits(expert_d2, torch.ones(expert_d2.size()).to(self.device))
            policy_loss1 = F.binary_cross_entropy_with_logits(policy_d1, torch.zeros(policy_d1.size()).to(self.device))
            policy_loss2 = F.binary_cross_entropy_with_logits(policy_d2, torch.zeros(policy_d2.size()).to(self.device))

            gail_loss = expert_loss1 + expert_loss2 + policy_loss1 + policy_loss2
            grad_pen = self.compute_grad_pen_mixture_2envs(expert_state1, expert_action1, expert_state2, expert_action2,
                                                           policy_state1, policy_action1, policy_state2, policy_action2)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n



    def compute_grad_pen_mixture_3envs(self, expert_state1, expert_action1, expert_state2, expert_action2,
                                       expert_state3, expert_action3, policy_state1, policy_action1,
                                       policy_state2, policy_action2, policy_state3, policy_action3, lambda_=10):
        expert_state = torch.cat([expert_state1, expert_state2, expert_state3], dim=0)
        expert_action = torch.cat([expert_action1, expert_action2, expert_action3], dim=0)
        policy_state = torch.cat([policy_state1, policy_state2, policy_state3], dim=0)
        policy_action = torch.cat([policy_action1, policy_action2, policy_action3], dim=0)

        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_gp_mixture_3envs(self, expert_loader1, expert_loader2, expert_loader3, rollouts1, rollouts2, rollouts3, rms=None):
        self.train()

        policy_data_generator1 = rollouts1.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)
        policy_data_generator2 = rollouts2.feed_forward_generator(None, mini_batch_size=expert_loader2.batch_size)
        policy_data_generator3 = rollouts3.feed_forward_generator(None, mini_batch_size=expert_loader3.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, expert_batch3, policy_batch1, policy_batch2, policy_batch3 \
                in zip(expert_loader1, expert_loader2, expert_loader3, policy_data_generator1, policy_data_generator2, policy_data_generator3):
            policy_state1, policy_action1 = policy_batch1[0], policy_batch1[2]
            policy_state2, policy_action2 = policy_batch2[0], policy_batch2[2]
            policy_state3, policy_action3 = policy_batch3[0], policy_batch3[2]
            policy_d1 = self.trunk(torch.cat([policy_state1, policy_action1], dim=1))
            policy_d2 = self.trunk(torch.cat([policy_state2, policy_action2], dim=1))
            policy_d3 = self.trunk(torch.cat([policy_state3, policy_action3], dim=1))
            expert_state1, expert_action1 = expert_batch1
            expert_state2, expert_action2 = expert_batch2
            expert_state3, expert_action3 = expert_batch3
            expert_state1 = np.clip((expert_state1.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state1 = torch.FloatTensor(expert_state1).to(self.device)
            expert_action1 = expert_action1.to(self.device)
            expert_state2 = np.clip((expert_state2.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state2 = torch.FloatTensor(expert_state2).to(self.device)
            expert_action2 = expert_action2.to(self.device)
            expert_state3 = np.clip((expert_state3.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state3 = torch.FloatTensor(expert_state3).to(self.device)
            expert_action3 = expert_action3.to(self.device)
            expert_d1 = self.trunk(torch.cat([expert_state1, expert_action1], dim=1))
            expert_d2 = self.trunk(torch.cat([expert_state2, expert_action2], dim=1))
            expert_d3 = self.trunk(torch.cat([expert_state3, expert_action3], dim=1))

            expert_loss1 = F.binary_cross_entropy_with_logits(expert_d1, torch.ones(expert_d1.size()).to(self.device))
            expert_loss2 = F.binary_cross_entropy_with_logits(expert_d2, torch.ones(expert_d2.size()).to(self.device))
            expert_loss3 = F.binary_cross_entropy_with_logits(expert_d3, torch.ones(expert_d3.size()).to(self.device))
            policy_loss1 = F.binary_cross_entropy_with_logits(policy_d1, torch.zeros(policy_d1.size()).to(self.device))
            policy_loss2 = F.binary_cross_entropy_with_logits(policy_d2, torch.zeros(policy_d2.size()).to(self.device))
            policy_loss3 = F.binary_cross_entropy_with_logits(policy_d3, torch.zeros(policy_d3.size()).to(self.device))

            gail_loss = expert_loss1 + expert_loss2 + expert_loss3 + policy_loss1 + policy_loss2 + policy_loss3
            grad_pen = self.compute_grad_pen_mixture_3envs(expert_state1, expert_action1, expert_state2, expert_action2,
                                                           expert_state3, expert_action3, policy_state1, policy_action1,
                                                           policy_state2, policy_action2, policy_state3, policy_action3)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n



    def compute_grad_pen_mixture_4envs(self, expert_state1, expert_action1, expert_state2, expert_action2,
                                       expert_state3, expert_action3, expert_state4, expert_action4, policy_state1, policy_action1,
                                       policy_state2, policy_action2, policy_state3, policy_action3, policy_state4, policy_action4, lambda_=10):
        expert_state = torch.cat([expert_state1, expert_state2, expert_state3, expert_state4], dim=0)
        expert_action = torch.cat([expert_action1, expert_action2, expert_action3, expert_action4], dim=0)
        policy_state = torch.cat([policy_state1, policy_state2, policy_state3, policy_state4], dim=0)
        policy_action = torch.cat([policy_action1, policy_action2, policy_action3, policy_action4], dim=0)

        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_gp_mixture_4envs(self, expert_loader1, expert_loader2, expert_loader3, expert_loader4,
                               rollouts1, rollouts2, rollouts3, rollouts4, rms=None):
        self.train()

        policy_data_generator1 = rollouts1.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)
        policy_data_generator2 = rollouts2.feed_forward_generator(None, mini_batch_size=expert_loader2.batch_size)
        policy_data_generator3 = rollouts3.feed_forward_generator(None, mini_batch_size=expert_loader3.batch_size)
        policy_data_generator4 = rollouts4.feed_forward_generator(None, mini_batch_size=expert_loader4.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, expert_batch3, expert_batch4, policy_batch1, policy_batch2, policy_batch3, policy_batch4 \
                in zip(expert_loader1, expert_loader2, expert_loader3, expert_loader4,
                       policy_data_generator1, policy_data_generator2, policy_data_generator3, policy_data_generator4):
            policy_state1, policy_action1 = policy_batch1[0], policy_batch1[2]
            policy_state2, policy_action2 = policy_batch2[0], policy_batch2[2]
            policy_state3, policy_action3 = policy_batch3[0], policy_batch3[2]
            policy_state4, policy_action4 = policy_batch4[0], policy_batch4[2]
            policy_d1 = self.trunk(torch.cat([policy_state1, policy_action1], dim=1))
            policy_d2 = self.trunk(torch.cat([policy_state2, policy_action2], dim=1))
            policy_d3 = self.trunk(torch.cat([policy_state3, policy_action3], dim=1))
            policy_d4 = self.trunk(torch.cat([policy_state4, policy_action4], dim=1))
            expert_state1, expert_action1 = expert_batch1
            expert_state2, expert_action2 = expert_batch2
            expert_state3, expert_action3 = expert_batch3
            expert_state4, expert_action4 = expert_batch4
            expert_state1 = np.clip((expert_state1.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state1 = torch.FloatTensor(expert_state1).to(self.device)
            expert_action1 = expert_action1.to(self.device)
            expert_state2 = np.clip((expert_state2.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state2 = torch.FloatTensor(expert_state2).to(self.device)
            expert_action2 = expert_action2.to(self.device)
            expert_state3 = np.clip((expert_state3.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state3 = torch.FloatTensor(expert_state3).to(self.device)
            expert_action3 = expert_action3.to(self.device)
            expert_state4 = np.clip((expert_state4.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state4 = torch.FloatTensor(expert_state4).to(self.device)
            expert_action4 = expert_action4.to(self.device)
            expert_d1 = self.trunk(torch.cat([expert_state1, expert_action1], dim=1))
            expert_d2 = self.trunk(torch.cat([expert_state2, expert_action2], dim=1))
            expert_d3 = self.trunk(torch.cat([expert_state3, expert_action3], dim=1))
            expert_d4 = self.trunk(torch.cat([expert_state4, expert_action4], dim=1))

            expert_loss1 = F.binary_cross_entropy_with_logits(expert_d1, torch.ones(expert_d1.size()).to(self.device))
            expert_loss2 = F.binary_cross_entropy_with_logits(expert_d2, torch.ones(expert_d2.size()).to(self.device))
            expert_loss3 = F.binary_cross_entropy_with_logits(expert_d3, torch.ones(expert_d3.size()).to(self.device))
            expert_loss4 = F.binary_cross_entropy_with_logits(expert_d4, torch.ones(expert_d4.size()).to(self.device))
            policy_loss1 = F.binary_cross_entropy_with_logits(policy_d1, torch.zeros(policy_d1.size()).to(self.device))
            policy_loss2 = F.binary_cross_entropy_with_logits(policy_d2, torch.zeros(policy_d2.size()).to(self.device))
            policy_loss3 = F.binary_cross_entropy_with_logits(policy_d3, torch.zeros(policy_d3.size()).to(self.device))
            policy_loss4 = F.binary_cross_entropy_with_logits(policy_d4, torch.zeros(policy_d4.size()).to(self.device))

            gail_loss = expert_loss1 + expert_loss2 + expert_loss3 + expert_loss4 + policy_loss1 + policy_loss2 + policy_loss3 + policy_loss4
            grad_pen = self.compute_grad_pen_mixture_4envs(expert_state1, expert_action1, expert_state2, expert_action2,
                                                           expert_state3, expert_action3, expert_state4, expert_action4,
                                                           policy_state1, policy_action1, policy_state2, policy_action2,
                                                           policy_state3, policy_action3, policy_state4, policy_action4)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n



    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = -(1 - s + 1e-8).log()

            if torch.isnan(reward):
                print("There is a Nan reward")
                import pdb;pdb.set_trace()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)



class WeightShared_discriminator(nn.Module):
    def __init__(self, multi_num, input_dim, hidden_dim, device):
        super(WeightShared_discriminator, self).__init__()

        self.device = device
        self.multi_num = multi_num

        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.Tanh()).to(device)
        self.logit1 = nn.Linear(hidden_dim, 1).to(device)
        self.logit2 = nn.Linear(hidden_dim, 1).to(device)
        self.logit3 = nn.Linear(hidden_dim, 1).to(device)
        self.logit4 = nn.Linear(hidden_dim, 1).to(device)

        self.trunk.train()
        self.logit1.train()
        self.logit2.train()
        self.logit3.train()
        self.logit4.train()

        self.optimizer1 = torch.optim.Adam([{"params": self.trunk.parameters()},
                                           {"params": self.logit1.parameters()}])
        self.optimizer2 = torch.optim.Adam([{"params": self.trunk.parameters()},
                                            {"params": self.logit2.parameters()}])
        self.optimizer3 = torch.optim.Adam([{"params": self.trunk.parameters()},
                                            {"params": self.logit3.parameters()}])
        self.optimizer4 = torch.optim.Adam([{"params": self.trunk.parameters()},
                                            {"params": self.logit4.parameters()}])

        self.returns1 = None
        self.ret_rms1 = RunningMeanStd(shape=())
        self.returns2 = None
        self.ret_rms2 = RunningMeanStd(shape=())
        self.returns3 = None
        self.ret_rms3 = RunningMeanStd(shape=())
        self.returns4 = None
        self.ret_rms4 = RunningMeanStd(shape=())

    def select_logit(self, input_data, multi_index):
        if multi_index == 0:
            out = self.logit1(input_data)
        elif multi_index == 1:
            out = self.logit2(input_data)
        elif multi_index == 2:
            out = self.logit3(input_data)
        elif multi_index == 3:
            out = self.logit4(input_data)
        return out

    def compute_grad_pen(self, multi_index, expert_state, expert_action, policy_state, policy_action, lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data) #
        disc = self.select_logit(disc, multi_index)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_WSD_2env(self, rollouts, expert_loader1, expert_loader2, rms):
        def int_update(multi_index, policy_batch, expert_batch, rms):
            loss = 0

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))  #
            policy_d = self.select_logit(policy_d, multi_index)
            expert_state, expert_action = expert_batch
            expert_state = np.clip((expert_state.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))  #
            expert_d = self.select_logit(expert_d, multi_index)

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(multi_index, expert_state, expert_action, policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()

            if multi_index == 0:
                self.optimizer1.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer1.step()
            elif multi_index == 1:
                self.optimizer2.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer2.step()
            elif multi_index == 2:
                self.optimizer3.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer3.step()
            elif multi_index == 3:
                self.optimizer4.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer4.step()
            elif multi_index == 4:
                self.optimizer5.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer5.step()
            return loss

        self.train()
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, policy_batch in zip(expert_loader1, expert_loader2, policy_data_generator):
            loss1 = int_update(0, policy_batch, expert_batch1, rms)
            loss2 = int_update(1, policy_batch, expert_batch2, rms)

            loss += loss1
            loss += loss2
            n += 1
        return loss / n

    def update_WSD_3env(self, rollouts, expert_loader1, expert_loader2, expert_loader3, rms):
        def int_update(multi_index, policy_batch, expert_batch, rms):
            loss = 0

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))  #
            policy_d = self.select_logit(policy_d, multi_index)
            expert_state, expert_action = expert_batch
            expert_state = np.clip((expert_state.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))  #
            expert_d = self.select_logit(expert_d, multi_index)

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(multi_index, expert_state, expert_action, policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()

            if multi_index == 0:
                self.optimizer1.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer1.step()
            elif multi_index == 1:
                self.optimizer2.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer2.step()
            elif multi_index == 2:
                self.optimizer3.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer3.step()
            elif multi_index == 3:
                self.optimizer4.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer4.step()
            elif multi_index == 4:
                self.optimizer5.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer5.step()
            return loss

        self.train()
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, expert_batch3, policy_batch \
                in zip(expert_loader1, expert_loader2, expert_loader3, policy_data_generator):
            loss1 = int_update(0, policy_batch, expert_batch1, rms)
            loss2 = int_update(1, policy_batch, expert_batch2, rms)
            loss3 = int_update(2, policy_batch, expert_batch3, rms)

            loss += loss1
            loss += loss2
            loss += loss3
            n += 1
        return loss / n

    def update_WSD_4env(self, rollouts, expert_loader1, expert_loader2, expert_loader3, expert_loader4, rms):
        def int_update(multi_index, policy_batch, expert_batch, rms):
            loss = 0

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))  #
            policy_d = self.select_logit(policy_d, multi_index)
            expert_state, expert_action = expert_batch
            expert_state = np.clip((expert_state.numpy() - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))  #
            expert_d = self.select_logit(expert_d, multi_index)

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(multi_index, expert_state, expert_action, policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()

            if multi_index == 0:
                self.optimizer1.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer1.step()
            elif multi_index == 1:
                self.optimizer2.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer2.step()
            elif multi_index == 2:
                self.optimizer3.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer3.step()
            elif multi_index == 3:
                self.optimizer4.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer4.step()
            elif multi_index == 4:
                self.optimizer5.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer5.step()
            return loss

        self.train()
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=expert_loader1.batch_size)

        loss = 0
        n = 0
        for expert_batch1, expert_batch2, expert_batch3, expert_batch4, policy_batch \
                in zip(expert_loader1, expert_loader2, expert_loader3, expert_loader4, policy_data_generator):
            loss1 = int_update(0, policy_batch, expert_batch1, rms)
            loss2 = int_update(1, policy_batch, expert_batch2, rms)
            loss3 = int_update(2, policy_batch, expert_batch3, rms)
            loss4 = int_update(3, policy_batch, expert_batch4, rms)

            loss += loss1
            loss += loss2
            loss += loss3
            loss += loss4
            n += 1
        return loss / n

    def predict_reward(self, multi_index, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            d = self.select_logit(d, multi_index)
            s = torch.sigmoid(d)
            reward = -(1 - s + 1e-8).log()

            if torch.isnan(reward):
                print("There ia a Nan reward")
                import pdb;pdb.set_trace()

            if multi_index == 0:
                if self.returns1 is None:
                    self.returns1 = reward.clone()
                if update_rms:
                    self.returns1 = self.returns1 * masks * gamma + reward
                    self.ret_rms1.update(self.returns1.cpu().numpy())
                return reward / np.sqrt(self.ret_rms1.var[0] + 1e-8)
            elif multi_index == 1:
                if self.returns2 is None:
                    self.returns2 = reward.clone()
                if update_rms:
                    self.returns2 = self.returns2 * masks * gamma + reward
                    self.ret_rms2.update(self.returns2.cpu().numpy())
                return reward / np.sqrt(self.ret_rms2.var[0] + 1e-8)
            elif multi_index == 2:
                if self.returns3 is None:
                    self.returns3 = reward.clone()
                if update_rms:
                    self.returns3 = self.returns3 * masks * gamma + reward
                    self.ret_rms3.update(self.returns3.cpu().numpy())
                return reward / np.sqrt(self.ret_rms3.var[0] + 1e-8)
            elif multi_index == 3:
                if self.returns4 is None:
                    self.returns4 = reward.clone()
                if update_rms:
                    self.returns4 = self.returns4 * masks * gamma + reward
                    self.ret_rms4.update(self.returns4.cpu().numpy())
                return reward / np.sqrt(self.ret_rms4.var[0] + 1e-8)



class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=0, subsample_frequency=1):
        all_trajectories = torch.load(file_name)

        self.obs_shape = all_trajectories['states'].size(2)
        self.acs_shape = all_trajectories['actions'].size(2)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        if num_trajectories == 0:
            idx = perm
            num_trajectories = all_trajectories['states'].size(0)
        else:
            idx = perm[:num_trajectories]

        if subsample_frequency < 2:
            subsample_frequency = 1

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        if subsample_frequency == 0:
            start_idx = 0
        else:
            start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
        print("Mean of returns: ", torch.mean(all_trajectories['rewards'].sum(axis=1)))
        print("Std. of returns: ", torch.std(all_trajectories['rewards'].sum(axis=1)))
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]

    def get_all_data(self):
        return self.trajectories['states'].detach().view(-1, self.obs_shape), self.trajectories['actions'].detach().view(-1, self.acs_shape)


