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


# TODO important i've disabled gradient descent for the moment



LOAD_SAVE = True
#if not LOAD_SAVE:
#  tf.config.run_functions_eagerly(True)
#else:
#  # TODO figure out graph mode
#  # seems like eager is default for me, have to manually disable
#  #tf.compat.v1.disable_eager_execution() # TF2; above holds
#  pass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import agent

OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'

CLIP_REWARDS = True
LR = 0.000001 #use this for fully connected
#LR = 0.001 # this is tf default
#LR = 0.01
#LR = 0.0001
#LR = 0.00001 
LR = 0.003

# FOR learning constant function = 1, best bet is rmsprop 0.003,
# and I was using l2 loss

#LR = 0.005 works quite well with RMSprop at learning the constant
# 1 function, but there are spurious
# spikes...

# SGD, 0.001, BS 128, ENVS1,, SBT 128*128, PUPC=2000, Reg=0
# gets to ~16 acc. 
# LR=0.0001 also gets there but takes much longer, doesn't improve after
#LR = 0.01 diverges sooner (best ~25), and weights explode

# NOTE: when printing out the distances for adjacent states,
# the matrix changes A LOT in between each step (~2). Lower learning
# rate significantly?

SAVE_CYCLES = 1
BATCH_SIZE = 128
ENVS = 1
STEPS_BETWEEN_TRAINING = 64
PARAM_UPDATES_PER_CYCLE = 8

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

    def print_accurracy(batch_size):
      coords = np.random.randint(0,agent.WIDTH,size=(batch_size, 2, 2, 1))
      #obs = np.zeros((batch_size, 2, agent.WIDTH, agent.HEIGHT))
      #obs[np.arange(batch_size), 0, coords[:,0,0], coords[:,0,1]] = 1
      #obs[np.arange(batch_size), 1, coords[:,1,0], coords[:,1,1]] = 1
      truths = np.sum(np.abs(coords[:,0,:] - coords[:,1,:]), axis=-2)
      #obs = tf.cast(obs, tf.float32)
      #obs = tf.expand_dims(obs, -1)
      #enc1 = actor.encode(obs[:,0,:,:,:])
      #enc2 = actor.encode(obs[:,1,:,:,:])
      enc1 = actor.encode(coords[:,0,:])
      enc2 = actor.encode(coords[:,1,:])
      #print(enc1)
      #print(enc2)
      dists = actor.distance(enc1, enc2)
      dists = tf.reduce_min(dists, axis=-1)
      #print(dists)
      #print(truths)
      truths = tf.cast(tf.squeeze(truths), tf.float32)
      #print(dists)
      print('accurracy: ' + str(tf.reduce_mean(tf.abs(dists - truths))))

    
    
    
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
        states_a, states_b, states_k = [s[:,:,:,0] for s in [states_a, states_b, states_k]] # remove time dimension
        # TODO should we even have parallel environments here?
        #states_a, states_b, states_k = [tf.reshape(x, (ENVS * BATCH_SIZE, agent.WIDTH, agent.HEIGHT, 1)) for x in [states_a, states_b, states_k]]
        states_a, states_b, states_k = [tf.reshape(x, (ENVS * BATCH_SIZE, 2, 1)) for x in [states_a, states_b, states_k]]
        enca = actor.encode(states_a)
        encb = actor.encode(states_b)
        #print(actor.distance(enca, encb))
    
        actions_a = tf.reshape(actions_a, (ENVS * BATCH_SIZE,))

        with tf.GradientTape(watch_accessed_variables=True) as tape:
          loss = actor.loss(states_a, states_b, states_k, actions_a)
          #print("{:6f}, {:6f}, {:6f}".format(loss_pve[0].numpy(), loss_pve[1].numpy(), loss_pve[2].numpy()))
          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)
        grad = tape.gradient(loss, actor.vars)
        # TODO add this back!
        agentOpt.apply_gradients(zip(grad, actor.vars))
        agent_losses += [loss]
        print_accurracy(100)
        #print_accurracy(3)

        print(loss_str)
      print(loss_str)


         
    
      cycle += 1
      if not cycle % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(actor, agent.model_savepath)
        with open(agent.loss_savepath, "wb") as fp:
          pickle.dump(agent_losses, fp)
        with open(agent.rewards_savepath, "wb") as fp:
          pickle.dump(episode_rewards, fp)

      
  
    
  
  
  
