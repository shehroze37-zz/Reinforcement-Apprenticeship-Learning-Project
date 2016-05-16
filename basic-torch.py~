#!/usr/bin/python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

from random import choice
from time import sleep
from time import time

import itertools as it


import lua
import numpy as np


torch = lua.require('torch')
lua.require('trepl')
dqn = lua.eval("dofile('dqn/NeuralQLearner.lua')")
tt = lua.eval("dofile('dqn/TransitionTable.lua')")



# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()
screen_width = 320
screen_height = 240
color_palette = 24

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
#game.load_config("../../examples/config/basic.cfg")

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game.set_vizdoom_path("../../bin/vizdoom")

# Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
game.set_doom_game_path("../../scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.

# Sets path to additional resources iwad file which is basically your scenario iwad.
# If not specified default doom2 maps will be used and it's pretty much useles... unless you want to play doom.
game.set_doom_scenario_path("../../scenarios/basic.wad")

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
#game.set_screen_resolution(ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.RGB24)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)

# Adds buttons that will be allowed. 
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(200)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(True)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()


# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.	
actions = [[True,False,False],[False,True,False],[False,False,True]]


actions_num = game.get_available_buttons_size()
complete_action_list = []


possible_actions_2 = lua.toTable({
        1: 1,
        2: 2,
        3: 3,
    })

dqn_args = {
    'target_q': 10000,
    'ncols': 1,
    'replay_memory': 1000000,
    'min_reward': -6,
    'max_reward': 100,
    'discount': 0.99,
    'learn_start': 50000,
    'hist_len': 4,
    'ep': 1,
    'network': "convnet_atari",
    'preproc': "preproc_atari",
    'gpu': -1,
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
    'actions': possible_actions_2,
    'verbose': 2,
    'TransitionTable':tt.TransitionTable

}

input_dims = lua.toTable({ })
input_dims[0] = dqn_args['ncols'] * dqn_args['hist_len']
input_dims[1] = screen_width 
input_dims[2] = screen_height
dqn_args['input_dims'] = input_dims 


agent = dqn.NeuralQLearner(dqn_args)


# Run this many episodes

episodes = 20000

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
sleep_time = 0.028
running_time = 0
for i in range(episodes):
    print("Episode #" + str(i+1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()
    t = 1

    action_index = 1

    while not game.is_episode_finished():

        # Gets the state

        t += 1
        game_observation = game.get_state()
        observation_2 = game_observation.image_buffer

        # Makes a random action and get remember reward.
        reward = game.make_action(actions[action_index - 1])
        #reward = game.make_action(choice(actions))

        game_state_after_action = game.get_state()
        observation = game_state_after_action.image_buffer


        game_state_after_action = np.ascontiguousarray(game_state_after_action)



        # Prints state's game variables. Printing the image is quite pointless.
        print("State #" + str(game_observation.number))
        print("Game variables:", game_observation.game_variables[0])
        print("Reward:", reward)
        print("=====================")

        action_index = agent.perceive(agent, reward, observation, False)


        if sleep_time>0:
            sleep(sleep_time)

    reward = game.get_total_reward()        

    #action_index = agent.perceive(agent, reward, observation, true)        
    
    running_time = t + 1

    if i%2000== 0:
        print("Episode finished with {} total timesteps " . format(running_time / 2000))
        #agent.report(agent)
        running_time = 0
        print("total reward:", game.get_total_reward())
        print("************************")
        #torch.save("out/new-" + str(i) + ".t7", agent)

    # Check how the episode went.
    #print("Episode finished with {} total timesteps " . format(running_time / 2000))
    #agent.report(agent)
    #running_time = 0
    #print("total reward:", game.get_total_reward())
    #print("************************")
    #torch.save("out/new-" + str(i) + ".t7", agent)


game.close()
