
import itertools as it
import pickle
from random import sample, randint, random
from time import time
from vizdoom import *

import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

from random import choice

# Q-learning settings:
replay_memory_size = 10000
discount_factor = 1
start_epsilon = float(1.0)
end_epsilon = float(0.1)
epsilon = start_epsilon
static_epsilon_steps = 4000
epsilon_decay_steps = 100000
epsilon_decay_stride = (start_epsilon - end_epsilon) / epsilon_decay_steps

# Max reward is about 100 (for killing) so it'll be normalized
reward_scale = 0.01

# Some of the network's and learning settings:
learning_rate = 0.00001
batch_size = 64
epochs = 200
training_steps_per_epoch = 5000
test_episodes_per_epoch = 100

# Other parameters
skiprate = 10
downsampled_x = 120
downsampled_y = 45
episodes_to_watch = 10

# Where to save and load network's weights.
params_savefile = "basic_params"
params_loadfile = None

# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img


# Replay memory:
class ReplayMemory:
    def __init__(self, capacity):

        state_shape = (capacity, 1, downsampled_y, downsampled_x)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0

        self.memSize = 4


        self.recent_s = np.zeros((4,1,downsampled_y,downsampled_x), dtype=np.float32)
        self.recent_a =  np.zeros(self.memSize, dtype=np.int32)
        self.recent_t =  np.zeros(self.memSize, dtype=np.bool_)

        self.s_size = downsampled_x * downsampled_y * 4

        self.bufferSize = 512
        self.recentMemSize = 4

        self.buf_a      = np.zeros(self.bufferSize, dtype=np.int32)
        self.buf_r      = np.zeros(self.bufferSize, dtype=np.float32)
        self.buf_term   = np.zeros(self.bufferSize, dtype=np.bool_)
        self.buf_s      = np.zeros((self.bufferSize, 4, downsampled_y, downsampled_x), dtype=np.float32)
        self.buf_s2     = np.zeros((self.bufferSize, 4, downsampled_y, downsampled_x), dtype=np.float32)


        self.buf_index = None
        self.histSpacing = 1
        self.numEntries = 0
        self.recentCount = 0
        self.zeroFrames = 1
        self.histLen = 4

        self.histIndices = {}

        for i in range(0,4):
            self.histIndices[i] = i * self.histSpacing

    def fill_buffer(self):
        if self.numEntries < self.bufferSize:
            print('Error : #Entries < Buffer Size' )
        else:
            self.buf_index = 0
            for i in range(0,self.bufferSize):
                s1, a,r,s2, term = self.sample_one(1)

                self.buf_s[i] = s1
                self.buf_s2[i] = s2
                self.buf_a[i] = a
                self.buf_r[i] = r
                self.buf_term[i] = term


    def sample_one(self, total):

        if self.numEntries > 1:
            index = None
            valid = False

            while not valid:
                
                index = sample(range(1,self.numEntries - self.recentMemSize), 1)
                if self.nonterminal[index[0] + self.recentMemSize - 1] == True:
                    valid = True

            return self.get(index[0])       
        else:
            print('Error : num entries < 1')    


    def get(self, index):
        s = self.concat_frames(index)
        s2 = self.concat_frames(index+1)
        ar_index = index + self.recentMemSize - 1 

        return s, self.a[ar_index], self.r[ar_index], s2, self.nonterminal[ar_index + 1]            


    def add_transition(self, s1, action, s2, reward):

        if self.numEntries < self.capacity:
            self.numEntries = self.numEntries + 1


        self.s1[self.oldest_index, 0] = s1
        if s2 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            self.s2[self.oldest_index, 0] = s2
            self.nonterminal[self.oldest_index] = True
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)



        #add the state to recent as well
        if self.recentCount >= 3:
            self.recent_s[0,0] = self.recent_s[2,0]
            self.recent_s[1,0] = self.recent_s[3,0]

            self.recent_s[2,0] = s1
            self.recent_s[3,0] = s2

        else:
            self.recent_s[self.recentCount,0] = s1
            self.recentCount = self.recentCount + 1

            self.recent_s[self.recentCount,0] = s2
            self.recentCount = self.recentCount + 1



    def concat_frames(self, index):

        s = self.s1
        nonterm = self.nonterminal

        state_shape = (1, 4, downsampled_y, downsampled_x)
        full_state = np.zeros(state_shape, dtype=np.float32)

        zero_out = False
        episode_start = 4

        for i in range(2,0,-1):
            if not zero_out:
                for j in range(index+self.histIndices[i] - 1, index + self.histIndices[i+1] - 2):
                    if nonterm[j] == False:
                        print('Zero out is true')
                        zero_out = True
                        break

            if zero_out == False:
                episode_start = i

        if self.zeroFrames == 0:
            episode_start = 0


        episode_start = 0
        #print('Episode start = ' + str(episode_start))    

        for i in range(episode_start, self.histLen):
            #print('concatenating frames')
            full_state[0,i] = s[index+self.histIndices[i]-1]    

        return full_state    
            

    def get_sample(self, sample_size):

        #fill the buffers and returns those buffers
        if self.buf_index == None or self.buf_index + sample_size - 1 > 512:
            self.fill_buffer()

        index = self.buf_index
        self.buf_index  = self.buf_index + sample_size

        buf_s = self.buf_s
        buf_s2 = self.buf_s2
        buf_a = self.buf_a
        buf_term = self.buf_term
        buf_r = self.buf_r



        #i = sample(range(0, self.size), sample_size)
        i = range(index, index + sample_size )
        #return self.s1[i], self.s2[i], self.a[i], self.r[i], self.nonterminal[i]
        return buf_s[i], buf_s2[i], buf_a[i], buf_r[i], buf_term[i]


# Creates the network:
def create_network(available_actions_num):
    # Creates the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State best Q-Value")
    r = tensor.vector("Rewards")
    nonterminal = tensor.vector("Nonterminal", dtype="int8")

    # Creates the input layer of the network.
    dqn = InputLayer(shape=[None, 4, downsampled_y, downsampled_x], input_var=s1)

    # Adds 3 convolutional layers, each followed by a max pooling layer.
    dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))

    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[3, 3],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
    # Adds a single fully connected layer.
    dqn = DenseLayer(dqn, num_units=512, nonlinearity=rectify, W=GlorotUniform("relu"),
                     b=Constant(.1))

    # Adds a single fully connected layer which is the output layer.
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)

    # Theano stuff
    q = get_output(dqn)
    # Only q for the chosen actions is updated more or less according to following formula:
    # target Q(s,a,t) = r + gamma * max Q(s2,_,t+1)
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * nonterminal * q2)
    loss = squared_error(q, target_q).mean()

    # Updates the parameters according to the computed gradient using rmsprop.
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    # Compiles theano functions
    print "Compiling the network ..."
    function_learn = theano.function([s1, q2, a, r, nonterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print "Network compiled."

    # Returns Theano objects for the net and functions.
    # We wouldn't need the net anymore but it is nice to save your model.
    return dqn, function_learn, function_get_q_values, function_get_best_action


# Creates and initializes the environment.
print "Initializing doom..."
game = DoomGame()
game.load_config("../../examples/config/health_gathering.cfg")
game.init()
print "Doom initialized."

# Creates all possible actions.
n = game.get_available_buttons_size()
actions = []
for perm in it.product([0, 1], repeat=n):
    actions.append(list(perm))

# Creates replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)
net, learn, get_q_values, get_best_action = create_network(len(actions))

# Loads the  network's parameters if the loadfile was specified
if params_loadfile is not None:
    params = pickle.load(open(params_loadfile, "r"))
    set_all_param_values(net, params)


# Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
def perform_learning_step():
    # Checks the state and downsamples it.
    s1 = convert(game.get_state().image_buffer)

    # With probability epsilon makes a random action.
    if random() <= epsilon or memory.numEntries < 4:
        a = randint(0, len(actions) - 1)
    else:
        # Chooses the best action according to the network.
        #a = get_best_action(s1.reshape([1, 1, downsampled_y, downsampled_x]))

        state_shape = (1, 4, downsampled_y, downsampled_x)
        all_states = np.zeros(state_shape, dtype=np.float32)

        all_states[0,0] = memory.recent_s[0,0]
        all_states[0,1] = memory.recent_s[1,0]
        all_states[0,2] = memory.recent_s[2,0]
        all_states[0,3] = memory.recent_s[3,0]

        a = get_best_action(all_states.reshape([1, 4, downsampled_y, downsampled_x]))



    reward = game.make_action(actions[a], skiprate + 1)
    reward *= reward_scale

    if game.is_episode_finished():
        s2 = None
    else:
        s2 = convert(game.get_state().image_buffer)
    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, reward)

    # Gets a single, random minibatch from the replay memory and learns from it.
    if memory.size > 4 * batch_size and memory.numEntries >= memory.bufferSize :
        s1, s2, a, reward, nonterminal = memory.get_sample(batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        print('Calculated Q values')
        loss = learn(s1, q2, a, reward, nonterminal)
        print('Calculated loss')
    else:
        loss = 0
    return loss


print "Starting the training!"

steps = 0
for epoch in range(epochs):
    print "\nEpoch", epoch
    train_time = 0
    train_episodes_finished = 0
    train_loss = []
    train_rewards = []

    train_start = time()
    print "\nTraining ..."
    game.new_episode()
    for learning_step in tqdm(range(training_steps_per_epoch)):
        # Learning and action is here.
        train_loss.append(perform_learning_step())
        print('Performed learning step')
        # I
        if game.is_episode_finished():
            r = game.get_total_reward()
            train_rewards.append(r)
            game.new_episode()
            train_episodes_finished += 1

        steps += 1
        if steps > static_epsilon_steps:
            epsilon = max(end_epsilon, epsilon - epsilon_decay_stride)

    train_end = time()
    train_time = train_end - train_start
    mean_loss = np.mean(train_loss)

    print train_episodes_finished, "training episodes played."
    print "Training results:"

    train_rewards = np.array(train_rewards)

    print "mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "epsilon:", epsilon
    print "t:", str(round(train_time, 2)) + "s"

    # Testing
    test_episode = []
    test_rewards = []
    test_start = time()
    print "Testing..."
    for test_episode in tqdm(range(test_episodes_per_epoch)):
        game.new_episode()
        while not game.is_episode_finished():
            state = convert(game.get_state().image_buffer).reshape([1, 1, downsampled_y, downsampled_x])
            best_action_index = get_best_action(state)

            game.make_action(actions[best_action_index], skiprate + 1)
        r = game.get_total_reward()
        test_rewards.append(r)

    test_end = time()
    test_time = test_end - test_start
    print "Test results:"
    test_rewards = np.array(test_rewards)
    print "mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min()
    print "t:", str(round(test_time, 2)) + "s"

    if params_savefile:
        print "Saving network weigths to:", params_savefile
        pickle.dump(get_all_param_values(net), open(params_savefile, "w"))
    print "========================="

print "Training finished! Time to watch!"

game.close()
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

# Sleeping time between episodes, for convenience.
episode_sleep = 0.5

for i in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = convert(game.get_state().image_buffer).reshape([1, 1, downsampled_y, downsampled_x])
        best_action_index = get_best_action(state)
        game.set_action(actions[best_action_index])
        for i in range(skiprate+1):
            game.advance_action()

    sleep(episode_sleep)
    r = game.get_total_reward()
    print "Total reward: ", r


def unit_tests():
    state_shape = (1, 4, downsampled_y, downsampled_x)
    all_states = np.zeros(state_shape, dtype=np.float32)


    s1 = convert(game.get_state().image_buffer)
    r1 = game.make_action(choice(actions), skiprate + 1)


    s2 = convert(game.get_state().image_buffer)
    r1 = game.make_action(choice(actions), skiprate + 1)



    s3 = convert(game.get_state().image_buffer)
    r1 = game.make_action(choice(actions), skiprate + 1)



    s4 = convert(game.get_state().image_buffer)
    r1 = game.make_action(choice(actions), skiprate + 1)

    all_states[0,0] = s1
    all_states[0,1] = s2
    all_states[0,2] = s3
    all_states[0,3] = s4

    all_states.reshape([1, 4, downsampled_y, downsampled_x])

    a = get_best_action(all_states.reshape([1, 4, downsampled_y, downsampled_x]))

    print(a)

    buffer_states    = np.zeros((512,4,45,120), dtype=np.float32)


    for i in range(0,512):
        buffer_states[i] = all_states

    i = range(4, 5)
    new_buffer = buffer_states[i]
    print(all_states.shape)



    print('Reshaping')
    s1.reshape([1, 4 ,downsampled_y, downsampled_x])


    print('New buffer shape')
    print(new_buffer.shape)

    q2 = np.max(get_q_values(new_buffer), axis=1)
    print(q2)
    
    game.close()