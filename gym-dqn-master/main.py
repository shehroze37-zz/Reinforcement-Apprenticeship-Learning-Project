import gym
import lua
import numpy as np

torch = lua.require('torch')
lua.require('trepl')
lua.require('cunn') # We run the network on GPU
dqn = lua.eval("dofile('dqn/NeuralQLearner.lua')")
tt = lua.eval("dofile('dqn/TransitionTable.lua')")

env = gym.make('Breakout-v0')

n_episode = 20000 # Original paper: 50000000

possible_actions = lua.toTable({
        1: 0,
        2: 2,
        3: 3,
        4: 1
    })
input_dims = lua.toTable({
        1: env.observation_space.shape[2],
        2: env.observation_space.shape[0],
        3: env.observation_space.shape[1],
    })

dqn_args = {
    'target_q': 10000,
    'ncols': 1,
    'replay_memory': 1000000,
    'min_reward': -1,
    'max_reward': 1,
    'discount': 0.99,
    'learn_start': 50000,
    'hist_len': 4,
    'ep': 1,
    'network': "convnet_atari",
    'preproc': "preproc_atari",
    'gpu': 0,
    'n_replay': 1,
    'clip_delta': 1,
    'valid_size': 500,
    'lr': 0.00025,
    'bufferSize': 512,
    'update_freq': 4,
    'minibatch_size': 32,
    'rescale_r': 1,
    'ep_end': 0.1,
    'state_dim': 7056,
    'actions': possible_actions,
    'verbose': 2,
    'TransitionTable':tt.TransitionTable,
}

agent = dqn.NeuralQLearner(dqn_args)
print "Created"

env.monitor.start('/data/gym/breakout/1')
running_t = 0
for i_episode in xrange(n_episode):
    observation = env.reset()
    action_index = 4
    done = False
    t=1
    while True:
        t += 1
        #env.render()
        observation, reward, done, info = env.step(possible_actions[action_index])
        observation = np.ascontiguousarray(observation.transpose([2,0,1]))
        if done:
            reward = -100
        action_index = agent.perceive(agent, reward, observation, done)
        if done:
            #print "Episode finished after {} timesteps".format(t+1)
            running_t += t+1
            break
    if i_episode%2000== 0:
        print "Episode finished after {} timesteps".format(running_t/2000)
        agent.report(agent)
        running_t = 0
        torch.save("out/net-" + str(i_episode) + ".t7", agent)

env.monitor.close()
