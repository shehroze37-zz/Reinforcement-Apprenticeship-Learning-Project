import gym
import lua
import numpy as np

torch = lua.require('torch')
lua.require('trepl')
lua.require('cunn') # We run the network on GPU
dqn = lua.eval("dofile('dqn/NeuralQLearner.lua')")
tt = lua.eval("dofile('dqn/TransitionTable.lua')")
lua.execute("dofile('dqn/Scale.lua')") # for the preproc

env = gym.make('Breakout-v0')

possible_actions = lua.toTable({
        1: 0,
        2: 2,
        3: 3,
        4: 1
    })

agent = torch.load("out/net-10000.t7")
agent.bestq = 0

observation = env.reset()
action_index = 4
done = False
t=1
while True:
    t += 1
    env.render()
    observation, reward, done, info = env.step(possible_actions[action_index])
    observation = np.ascontiguousarray(observation.transpose([2,0,1]))
    action_index = agent.perceive(agent, reward, observation, done, True)
    if done:
        print "Episode finished after {} timesteps".format(t+1)
        break
