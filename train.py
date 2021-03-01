import tensorflow as tf
from replay import PrioritizedReplayBuffer
import timeit
from queue import Queue
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from toyenv import ToyEnv, LoopEnv
import random
import pickle
import os
from skimage.transform import resize
from tensorflow.keras import Model
import gym
from contextlib import ExitStack
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent
from test import actionAccuracy, printAccuracy

LOAD_SAVE = True

''' 
with alpha = 3 beta = 0, can use higher learning rate, and doesn't diverge (will temorarily diverge but always goes back)

'''

OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'

BUFFER_SIZE = 2**13
#BUFFER_SIZE = 8

CLIP_REWARDS = True
#LR = 0.000001 #use this for fully connected
#LR = 0.001 # this is tf default
#LR = 0.01
#LR = 0.0001
#LR = 0.00001 
LR = 0.000003 # this is the best so far for toyenv, diverges at an order of magnitude higher.
LR = 0.00003 # diverges here! UNLESS we use experience replay and then it seems to work ok
#LR = 0.0003 
#LR = 0.0003

USE_TARGET = False

SAVE_CYCLES = 1
BATCH_SIZE = 128
ENVS = 2
STEPS_BETWEEN_TRAINING = 512
PARAM_UPDATES_PER_CYCLE = 500

TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP  = 20

''' without replay_buffer.updateWeights, get to 95% acc in 25k updates '''


'''


Here's a possible explanation for explosions:
if we are collecting data, let's say the last states we visit are a,a,a. And we 
never see a transition away from a. Then every time we train on a -> a, d(a,x) 
is being increased (if the training affects d(a,_,action that would move closer to x)
basically, we just need one incorrect value in the network for a transition that wasn't
observed, and then if the training on other spots increases this value along with
the other observed transitions, then it can explode. This would explain why more
PARAM_UPDATES_PER_CYCLE is bad, but not why more STEPS_BETWEEN_TRAINING is bad... 
'''


if __name__ == '__main__':

  #sess = tf.compat.v1.Session()
  #with sess.as_default():
  if True:

    with open(agent.loss_savepath, "rb") as f: 
      agentloss = pickle.load(f)
    with open(agent.accs_savepath, "rb") as f: 
      accs = pickle.load(f)
    with open(agent.rewards_savepath, "rb") as f:
      episode_rewards = pickle.load(f)
    
    if (LOAD_SAVE):
      #actor = tf.saved_model.load(agent.model_savepath)
      actor = tf.saved_model.load(agent.model_savepath)
      target_actor = agent.Agent() if USE_TARGET else actor
        
      #sess.run(tf.compat.v1.global_variables_initializer())
    else:
      tf.config.run_functions_eagerly(True) 
      actor = agent.Agent()
      target_actor = agent.Agent() if USE_TARGET else actor
    
    if OPTIMIZER == 'Adam':
      agentOpt = tf.keras.optimizers.Adam(learning_rate = LR)
    elif OPTIMIZER == 'RMSprop':
      agentOpt = tf.keras.optimizers.RMSprop(learning_rate = LR)
    elif OPTIMIZER == 'SGD':
      agentOpt = tf.keras.optimizers.SGD(learning_rate = LR)

    def copyWeightsFromTo(actor, target):
      for i,v in enumerate(actor.vars):
        target.vars[i].assign(v)

    if USE_TARGET:
      copyWeightsFromTo(actor, target_actor)
    
    maxtd = 0
    envs = []
    states = []
    statelists = []

    # make environment
    for i in range(ENVS):
 
      #env = gym.make(agent.ENVIRONMENT)
      env = ToyEnv(agent.TOYENV_SIZE, agent.USE_COORDS)
      state1 = tf.cast(env.reset(), tf.float32)
      # 0 action is 'NOOP'
      state2 = tf.cast(env.step(0)[0], tf.float32)
      state3 = tf.cast(env.step(0)[0], tf.float32)
      state4 = tf.cast(env.step(0)[0], tf.float32)
      statelist = [state1, state2, state3, state4]
      # TODO: abstract away environment pre-processing
      #statelist = [tf.image.rgb_to_grayscale(s) for s in statelist]
      #statelist = [tf.image.resize(s,(84,110)) for s in statelist] #TODO does method of downsampling matter?
      #statelist = [s / 255.0 for s in statelist]
      state = tf.stack(statelist, -1)
      #state = tf.squeeze(state)
      envs += [env]
      states.append(state)
      statelists += [statelist]

    states = tf.stack(states)

    
    
    
    cycle = 0
    agent_losses = []
    total_rewards = [0 for _ in envs]


    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, 0.001)

    while True: 


      states_l, actions_l, rewards_l, dones_l = [],[],[],[]
      print('acting')

      if cycle == 0:
        cycle_steps = BUFFER_SIZE // TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP
      else:
        cycle_steps = STEPS_BETWEEN_TRAINING
      for step in range(cycle_steps):
        actions = np.random.randint(0,5,size=(ENVS,))
        next_states, rewards, dones = [], [], []

        for i in range(ENVS):
          #atend = envs[i].coords[1] == 83
          #print(envs[i].coords[1])
          #if atend: 
            #input('at edge')
            #print(actions[i])
          observation, reward, done, info = envs[i].step(actions[i])
          #if atend: input(observation)
          #if envs[i].coords[1] == 0 and atend: input('looped!')
          if CLIP_REWARDS:
            if reward > 1: reward = 1.0
            if reward < -1: reward = -1.0
          total_rewards[i] += reward

          #observation = tf.image.rgb_to_grayscale(observation)
          #observation = tf.image.resize(observation,(84,110)) 
          #observation = observation / 255.0
          observation = tf.cast(observation, tf.float32)
          if (done): 
            envs[i].reset()
            episode_rewards += [total_rewards[i]]
            print("Finished episode %d, reward: %f" % (len(episode_rewards), total_rewards[i]))
            total_rewards[i] = 0

          statelists[i] = statelists[i][1:]
          statelists[i].append(observation)
          next_state = tf.stack(statelists[i], -1)
          #next_state = tf.squeeze(next_state)
          next_states.append(next_state)
          dones.append(float(done))
          rewards.append(reward)

        # need to copy to CPU so we don't use all the GPU memory
        #with tf.device('/device:CPU:0'):
        if True:
          states_l.append(tf.identity(states))
          actions_l.append(tf.identity(actions))
          rewards_l.append(tf.stack(rewards))
          dones_l.append(tf.stack(dones))

        states = tf.stack(next_states)

      states_l.append(states)
      

      # compute value targets (discounted returns to end of episode (or end of training))
      #rewards_l[-1] += (1 - dones_l[-1]) * actor.policy_and_value(states)[1]
      with tf.device('/device:CPU:0'):
        for i in range(len(rewards_l)-2, -1, -1):
          rewards_l[i] = rewards_l[i] + agent.DISCOUNT * (1-dones_l[i]) * rewards_l[i+1]

      print("Frames: %d" % ((cycle + 1) * STEPS_BETWEEN_TRAINING * ENVS))
      print("Param updates: %d" % (cycle * PARAM_UPDATES_PER_CYCLE))

      '''
      TODO this is changed
      states_a, states_b, actions_a = [tf.stack(s) for s in [states_a, states_b, actions_a]]
      states_a, states_b, = [s[:,:,:,-1] for s in [states_a, states_b, states_k]] # remove time dimension
      states_a, states_b, = [tf.reshape(x, (ENVS * BATCH_SIZE, 2, 1)) for x in [states_a, states_b, states_k]]
      '''

      def dataGen():
        for i in range(len(states_l) - 1):
          for j in range(TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP):
            # TODO should pick states_k from all states seen, not just most recent iteration?
            states_k = states_l[random.randint(0,len(states_l) - 1)] 
            envs_data = (states_l[i], states_l[i+1], states_k, actions_l[i], rewards_l[i], dones_l[i])
            for e,_ in enumerate(envs):
              dp = [ed[e] for ed in envs_data]
              yield dp
            

      #replay_buffer.addDatapoints(dataGen(), [1 for _ in range((len(states_l) - 1) * TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP * ENVS)])
      replay_buffer.addDatapoints(dataGen())

      def getBatch():
        # TODO: there's an error here, doesn't take into account dones. Shouldn't matter for toy environment though.
        b,indices,probs = replay_buffer.sampleBatch(BATCH_SIZE)
        states_a, states_b, states_k, actions_a, _, _ = zip(*b)
        states_a, states_b, states_k, actions_a = [tf.stack(s) for s in [states_a, states_b, states_k, actions_a]]
        states_a, states_b, states_k = [s[...,-1:] for s in [states_a, states_b, states_k]] # remove time dimension
        states_a, states_b, states_k = [tf.reshape(x, [BATCH_SIZE] + agent.INPUT_SHAPE) for x in [states_a, states_b, states_k]]
        Dbk_target = target_actor.distance_states(states_b, states_k)
        return (states_a, states_b, states_k, actions_a, Dbk_target), indices, tf.cast(probs, tf.float32)
      

      print('training distance')
      for b in range(PARAM_UPDATES_PER_CYCLE):
        (states_a, states_b, states_k, actions_a, Dbk_target), indices, probs = getBatch()
        with tf.GradientTape(watch_accessed_variables=True) as tape:
          loss, td_error = actor.loss(states_a, states_b, states_k, actions_a, Dbk_target, probs)

          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)
        grad = tape.gradient(loss, actor.vars)
        agentOpt.apply_gradients(zip(grad, actor.vars))
        agent_losses += [loss]
        #printAccuracy(env, 100, actor)
        acc = actionAccuracy(env,100, actor)
        accs += [acc]
        # TODO add bck in
        # important to convert td_error to numpy first,
        # ovtherwise it takes forever
        replay_buffer.updateWeights(indices, td_error.numpy())
        maxtd = max(maxtd, tf.reduce_max(td_error))
        #print('MAX TD: ' + str(maxtd))

        print(loss_str)


      if USE_TARGET:
        copyWeightsFromTo(actor, target_actor)
         
    
      cycle += 1
      if not cycle % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(actor, agent.model_savepath)
        with open(agent.loss_savepath, "wb") as fp:
          pickle.dump(agent_losses, fp)
        with open(agent.rewards_savepath, "wb") as fp:
          pickle.dump(episode_rewards, fp)
        with open(agent.accs_savepath, "wb") as fp:
          pickle.dump(accs, fp)

      
  
    
  
  
  
