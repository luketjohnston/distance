import tensorflow as tf
import timeit
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from toyenv import ToyEnv
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
from test import *


'''
Note: the inaccurrate actions appear (from small amount of testing)
  to always be the no-op. (they are slightly smaller than the next
  best action). Also, 1-distances are always more than 1 (
  not sure why this is). Finally, training seems to diverge with
  rmsprop, I suspect its because of momentum failing when gradient
  is super small?" TODO this is something we can test
'''



LOAD_SAVE = True
#if not LOAD_SAVE:
#  tf.config.run_functions_eagerly(True)
#else:
#  # TODO figure out graph mode
#  # seems like eager is default for me, have to manually disable
#  #tf.compat.v1.disable_eager_execution() # TF2; above holds
#  pass


OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
#OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'

# it seems like l2 loss in agent is better for learning action,
# but ends up with worse average distance from correct

CLIP_REWARDS = True
#LR = 0.000001 #use this for fully connected
#LR = 0.001 # this is tf default
#LR = 0.01
#LR = 0.0001
#LR = 0.00001 
LR = 0.003
LR = 0.00003

# FOR learning constant function = 1, best bet is rmsprop 0.003,
# and I was using l2 loss

# Here's a possible explanation for explosions:
# if we are collecting data, let's say the last states we visit are a,a,a. And we 
# never see a transition away from a. Then every time we train on a -> a, d(a,x) 
# is being increased (if the training affects d(a,_,action that would move closer to x)
# basically, we just need one incorrect value in the network for a transition that wasn't
# observed, and then if the training on other spots increases this value along with
# the other observed transitions, then it can explode. This would explain why more
# PARAM_UPDATES_PER_CYCLE is bad, but not why more STEPS_BETWEEN_TRAINING is bad... 

SAVE_CYCLES = 1
BATCH_SIZE = 128
ENVS = 16
STEPS_BETWEEN_TRAINING = 128
PARAM_UPDATES_PER_CYCLE = 256

'''
These settings seem stable, 
BATCH_SIZE = 128
ENVS = 1
STEPS_BETWEEN_TRAINING = 64
PARAM_UPDATES_PER_CYCLE = 8
LR = 0.003
OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'
    loss += tf.reduce_mean(tf.pow(Dab_a - 1, 2))
    loss = tf.reduce_mean(tf.pow(Dak_a - target, 2))
    REGULARIZATION_WEIGHT = 0
'''


#STEPS_BETWEEN_TRAINING = 128
#PARAM_UPDATES_PER_CYCLE = 128



if __name__ == '__main__':

  #sess = tf.compat.v1.Session()
  #with sess.as_default():
  if True:

    with open(agent.loss_savepath, "rb") as f: 
      agentloss = pickle.load(f)
    with open(agent.rewards_savepath, "rb") as f:
      episode_rewards = pickle.load(f)
    
    if (LOAD_SAVE):
      #actor = tf.saved_model.load(agent.model_savepath)
      actor = tf.saved_model.load(agent.model_savepath)
      #sess.run(tf.compat.v1.global_variables_initializer())
    else:
      actor = agent.Agent()
    
    if OPTIMIZER == 'Adam':
      agentOpt = tf.keras.optimizers.Adam(learning_rate = LR)
    elif OPTIMIZER == 'RMSprop':
      agentOpt = tf.keras.optimizers.RMSprop(learning_rate = LR)
    elif OPTIMIZER == 'SGD':
      agentOpt = tf.keras.optimizers.SGD(learning_rate = LR)
    
    envs = []
    states = []
    statelists = []

    # make environment
    for i in range(ENVS):
 
      #env = gym.make(agent.ENVIRONMENT)
      env = ToyEnv(agent.WIDTH)
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
      state = tf.squeeze(state)
      envs += [env]
      states.append(state)
      statelists += [statelist]

    states = tf.stack(states)

    
    
    
    cycle = 0
    agent_losses = []
    total_rewards = [0 for _ in envs]

    #def sample_batch(replay_buffer, batch_size):
    #  sample = random.sample(replay_buffer, batch_size)
    #  return tf.stack(sample, 0)

    wgan_replay_buffer = []


    while True: 


      states_l, actions_l, old_action_probs_l, rewards_l, dones_l = [],[],[],[],[]
      print('acting')

      for step in range(STEPS_BETWEEN_TRAINING):
        actions = np.random.randint(0,5,size=(ENVS,))
        next_states, rewards, dones = [], [], []

        for i in range(ENVS):
          observation, reward, done, info = envs[i].step(actions[i])
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
          state = tf.stack(statelists[i], -1)
          state = tf.squeeze(state)
          next_states.append(state)
          dones.append(float(done))
          rewards.append(reward)

        # need to copy to CPU so we don't use all the GPU memory
        #with tf.device('/device:CPU:0'):
        if True:
          states_l.append(tf.identity(states))
          actions_l.append(tf.identity(actions))
          #old_action_probs_l.append(tf.identity(old_action_probs))
          rewards_l.append(tf.squeeze(tf.stack(rewards)))
          dones_l.append(tf.stack(dones))

        states = tf.stack(next_states)

      # compute value targets (discounted returns to end of episode (or end of training))
      #rewards_l[-1] += (1 - dones_l[-1]) * actor.policy_and_value(states)[1]
      with tf.device('/device:CPU:0'):
        for i in range(len(rewards_l)-2, -1, -1):
          rewards_l[i] = rewards_l[i] + agent.DISCOUNT * (1-dones_l[i]) * rewards_l[i+1]

      print("Frames: %d" % ((cycle + 1) * STEPS_BETWEEN_TRAINING * ENVS))
      print("Param updates: %d" % (cycle * PARAM_UPDATES_PER_CYCLE))

      

      print('training distance')
      indices = list(range(len(states_l)))
      for b in range(PARAM_UPDATES_PER_CYCLE):
        # TODO: there's an error here, doesn't take into account dones. Shouldn't matter for toy environment though.
        batch_inds_a = random.choices(indices[:-1], k=BATCH_SIZE)
        batch_inds_k = random.choices(indices, k=BATCH_SIZE)
        states_a = [states_l[i] for i in batch_inds_a]
        states_b = [states_l[i+1] for i in batch_inds_a]
        states_k = [states_l[i] for i in batch_inds_k]
        actions_a = [actions_l[i] for i in batch_inds_a]
        states_a, states_b, states_k, actions_a = [tf.stack(s) for s in [states_a, states_b, states_k, actions_a]]
        #states_a, states_b, states_k = [s[:,:,:,:,0] for s in [states_a, states_b, states_k]] # remove time dimension
        states_a, states_b, states_k = [s[:,:,:,-1] for s in [states_a, states_b, states_k]] # remove time dimension
        # TODO should we even have parallel environments here?
        #states_a, states_b, states_k = [tf.reshape(x, (ENVS * BATCH_SIZE, agent.WIDTH, agent.HEIGHT, 1)) for x in [states_a, states_b, states_k]]
        states_a, states_b, states_k = [tf.reshape(x, (ENVS * BATCH_SIZE, 2, 1)) for x in [states_a, states_b, states_k]]
        enca = actor.encode(states_a)
        encb = actor.encode(states_b)
        #print(actor.distance(enca, encb))
    
        actions_a = tf.reshape(actions_a, (ENVS * BATCH_SIZE,))

        with tf.GradientTape(watch_accessed_variables=True) as tape:
          #print(states_a[0,:])
          #print(states_b[0,:])
          #print(actions_a[0])
          #input('continue...')
          loss = actor.loss(states_a, states_b, states_k, actions_a)
          #print("{:6f}, {:6f}, {:6f}".format(loss_pve[0].numpy(), loss_pve[1].numpy(), loss_pve[2].numpy()))
          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)
        grad = tape.gradient(loss, actor.vars)
        # TODO add this back!
        agentOpt.apply_gradients(zip(grad, actor.vars))
        agent_losses += [loss]
        printAccuracy(100, actor)
        actionAccuracy(100, actor)

        print(loss_str)
      #print(loss_str)


         
    
      cycle += 1
      if not cycle % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(actor, agent.model_savepath)
        with open(agent.loss_savepath, "wb") as fp:
          pickle.dump(agent_losses, fp)
        with open(agent.rewards_savepath, "wb") as fp:
          pickle.dump(episode_rewards, fp)

      
  
    
  
  
  
