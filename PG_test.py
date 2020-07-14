N_EPISODES = 2

from atari_wrappers import get_env
from models import *
from utils import *

model_name = 'PG_A2C_02'
# model_name = 'ReinforceNaive_01'
env_id = "PongNoFrameskip-v4"
identity = env_id + '_' + model_name
env = get_env(env_id)
net = AtariA2C(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(identity + '.pth'))

for i_episode in range(N_EPISODES):
    total_reward = test_policy(net, env, True)
    print('Episode: %d total_reward: %.3f' % (i_episode, total_reward))
