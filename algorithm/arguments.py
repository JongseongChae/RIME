import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--update-algo', default='ppo')
    parser.add_argument('--algo-name', default=None, help='the name of algorithm we are going to use')
    parser.add_argument('--sampled-envs', type=int, default=0, help='N (the number of the sampled interaction envs.')
    parser.add_argument('--expert-path1', default=None)
    parser.add_argument('--expert-path2', default=None)
    parser.add_argument('--expert-path3', default=None)
    parser.add_argument('--expert-path4', default=None)
    parser.add_argument('--gail-batch-size', type=int, default=128, help='gail batch size')
    parser.add_argument('--gail-epoch', type=int, default=5, help='gail epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--use-gae', action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter')
    parser.add_argument('--entropy-coef', type=float, default=0, help='entropy term coefficient')
    parser.add_argument('--expert-entropy-coef', type=float, default=0, help='entropy term coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False)
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=2048)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--num-mini-batch', type=int, default=32)
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter')
    parser.add_argument('--num-env-steps', type=int, default=10000000, help='number of environment steps to train')
    parser.add_argument('--env-name', default=None, help='environment to train on')
    parser.add_argument('--env-parameter', default=None, help='environment to train on')
    parser.add_argument('--log-dir', default='./tmp/gym/', help='directory to save agent logs')
    parser.add_argument('--save-dir', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--use-proper-time-limits', action='store_false', default=True, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_false', default=True, help='use a linear schedule on the learning rate')
    parser.add_argument('--initial-dstep', type=int, default=100)
    parser.add_argument('--warmup-iters', type=int, default=10)
    parser.add_argument('--rms-iters', type=int, default=10)
    parser.add_argument('--no-rms', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    return args