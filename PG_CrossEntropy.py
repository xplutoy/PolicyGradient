from collections import namedtuple

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from models import PolicyNet
from utils import *

BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', ['observation', 'action'])

model_name = 'CrossEntropy'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name
env = gym.make(env_id)
net = PolicyNet(env.observation_space.shape[0], env.action_space.n)


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    while True:
        action = net.action_dist([obs]).sample().item()
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    train_obs_v = torch.FloatTensor(train_obs).to(DEVICE)
    train_act_v = torch.FloatTensor(train_act).to(DEVICE)
    return train_obs_v, train_act_v, reward_bound, reward_mean


# train
criterion = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=0.01)
writer = SummaryWriter(comment=identity)

for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
    obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

    trainer.zero_grad()
    action_scores_v = net(obs_v)
    loss_v = criterion(action_scores_v, acts_v.long())
    loss_v.backward()
    trainer.step()

    print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_bound", reward_b, iter_no)
    writer.add_scalar("reward_mean", reward_m, iter_no)
    if reward_m > 199:
        print("Solved!")
        torch.save(net.state_dict(), identity + '.pth')
        break

writer.close()
