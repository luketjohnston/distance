import tensorflow as tf
import code
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
PROFILE = True

''' 
with alpha = 3 beta = 0, can use higher learning rate, and doesn't diverge (will temorarily diverge but always goes back)

'''

#TOY_ENV, 20x20
#ALPHA = 2
#ALPHA = 0

USE_BUFFER = True
TEST_LIMITED_STATES = False


OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'Adam' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER_EPSILON = 1e-7 #1e-7 is default for adam
GRAD_CLIP = 0.01
REGULARIZATION_WEIGHT = 0.0

BUFFER_SIZE = 2**18 # 2**17 is max we can hold in memory it seems
#BUFFER_SIZE = 2**10
#BUFFER_SIZE = 128


CLIP_REWARDS = True
#LR = 0.000001 #use this for fully connected
#LR = 0.001 # this is tf default
LR = 0.000003 # this is the best so far for toyenv, diverges at an order of magnitude higher.
LR = 0.00003 # diverges here! UNLESS we use experience replay and then it seems to work ok
LR = 0.0001
LR = 0.00001
LR = 0.01 
LR = 0.0001  # works for 20x20 toyenv
LR = 0.00001  # works for 20x20 toyenv
#LR = 0.000001  # too low for 20x20 toyenv, log


USE_TARGET = False
SAVE_CYCLES = 1
BATCH_SIZE = 16 # use small batch size, otherwise sample_rollout dominates runtime.
BATCH_SIZE = 128
# TODO try to optimize replay_buffer, so can increase batch size. 
STEPS_BETWEEN_TRAINING = 512
PARAM_UPDATES_PER_CYCLE = 500
if TEST_LIMITED_STATES:
  PARAM_UPDATES_PER_CYCLE *= 999999
#TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP  = 20



if __name__ == '__main__':

  #sess = tf.compat.v1.Session()
  #with sess.as_default():
  if True:

    with open(agent.picklepath, "rb") as f: 
      save = pickle.load(f)


      accs = [] if not 'accs' in save else save['accs']
      episode_rewards = [] if not 'episode_rewards' in save else save['episode_rewards']
      max_grads = [] if not 'max_grads' in save else save['max_grads']
      max_weights = [] if not 'max_weights' in save else save['max_weights']
      agent_losses = [] if not 'agent_losses' in save else save['agent_losses']
    
    if (LOAD_SAVE):
      #actor = tf.saved_model.load(agent.model_savepath)
      actor = tf.saved_model.load(agent.model_savepath)
      target_actor = agent.Agent() if USE_TARGET else actor
        
      #sess.run(tf.compat.v1.global_variables_initializer())
    else:
      tf.config.run_functions_eagerly(True) 
      actor = agent.Agent()
      target_actor = agent.Agent() if USE_TARGET else actor
    
    opt_params = {'learning_rate': LR, 'clipvalue': GRAD_CLIP}
    if OPTIMIZER == 'Adam':
      opt = tf.keras.optimizers.Adam
      opt_params['epsilon'] = OPTIMIZER_EPSILON
    elif OPTIMIZER == 'RMSprop':
      opt = tf.keras.optimizers.RMSprop
      opt_params['epsilon'] = OPTIMIZER_EPSILON
    elif OPTIMIZER == 'SGD':
      opt = tf.keras.optimizers.SGD

    agentOpt = opt(**opt_params)
    if 'optimizer_state' in save:
      agentOpt = opt.from_config(save['opt_config'])
      agentOpt.set_weights(save['opt_weights'])

    def copyWeightsFromTo(actor, target):
      for i,v in enumerate(actor.vars):
        target.vars[i].assign(v)

    if USE_TARGET:
      copyWeightsFromTo(actor, target_actor)
    
    maxtd = 0
    states = []
    statelists = []

    # TODO: implement tf.data for optimization
    if not agent.TOY_ENV:
      def preprocess(s):
        s = tf.image.rgb_to_grayscale(s)
        s = tf.image.resize(s,(agent.INPUT_SHAPE[0],agent.INPUT_SHAPE[1]))
        s = tf.squeeze(s)
        return tf.cast(s / 255.0, tf.float32)
    else:
      def preprocess(s):
        return tf.cast(s, tf.float32)


    # make environment
    env = agent.makeEnv()
    state1 = tf.cast(env.reset(), tf.float32)
    # 0 action is 'NOOP'
    state2 = tf.cast(env.step(0)[0], tf.float32)
    state3 = tf.cast(env.step(0)[0], tf.float32)
    state4 = tf.cast(env.step(0)[0], tf.float32)
    statelist = [state1, state2, state3, state4]
    statelist = [preprocess(s) for s in statelist]
    state = tf.stack(statelist, -1)
    observation = statelist[-1] # most recent observation

    cycle = 0
    nanCount = 0 if not 'nanCount' in save else save['nanCount']
    total_rewards = 0
    

    saved_data_shape = tuple(agent.INPUT_SHAPE[:-1])
    data_type_and_shape = ((np.float32, saved_data_shape), (np.int32, ()), (np.float32, ()))
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, 0.001, data_type_and_shape)

    states_l  = statelist[:-1]
    actions_l = [0 for _ in statelist[:-1]]
    rewards_l = [0 for _ in statelist[:-1]]
    dones_l   = [0 for _ in statelist[:-1]]

    replay_buffer.addDatapoints(zip(zip(states_l, actions_l, rewards_l), dones_l))

    while True: 

      if cycle == 1 and PROFILE:
        tf.profiler.experimental.start('logdir')
      if cycle == 4 and PROFILE:
        tf.profiler.experimental.stop()
  

      if USE_BUFFER:
        states_l, actions_l, rewards_l, dones_l = [],[],[],[]
        print('acting')

        if cycle == 0:
          cycle_steps = BUFFER_SIZE
        else:
          cycle_steps = STEPS_BETWEEN_TRAINING
        for step in range(cycle_steps):
          action = random.randrange(0,agent.ACTIONS)

          #print(agent.ACTION_MAP[action])
          prev_observation = observation
          observation, reward, done, info = env.step(agent.ACTION_MAP[action])
          if CLIP_REWARDS:
            if reward > 1: reward = 1.0
            if reward < -1: reward = -1.0
          total_rewards += reward
          observation = preprocess(observation)

                

          statelist = statelist[1:]
          statelist.append(observation)
          next_state = tf.stack(statelist, -1)

          # need to copy to CPU so we don't use all the GPU memory
          with tf.device('/device:CPU:0'):
          #if True:
            states_l.append(tf.identity(prev_observation))
            actions_l.append(tf.identity(action))
            rewards_l.append(tf.identity(reward))
            dones_l.append(done)

          state = next_state
          if (done): 
            episode_rewards += [total_rewards]
            print("Finished episode %d, reward: %f" % (len(episode_rewards), total_rewards))
            total_rewards = 0

            statelist = []
            obs = preprocess(env.reset())
            rew = 0
            for i in range(agent.INPUT_SHAPE[-1]):
              statelist.append(obs)
              states_l.append(tf.identity(observation))
              actions_l.append(tf.identity(0))
              rewards_l.append(tf.identity(rew))
              if i == agent.INPUT_SHAPE[-1] - 1:
                break
              obs,rew,done,info = env.step(0)
              obs = preprocess(obs)
              assert(not done) # if environment ends in first few noops, we have a problem
              total_rewards += rew
            state = tf.stack(statelist, -1)


        # compute value targets (discounted returns to end of episode (or end of training))
        #rewards_l[-1] += (1 - dones_l[-1]) * actor.policy_and_value(states)[1]
        #with tf.device('/device:CPU:0'):
        #  for i in range(len(rewards_l)-2, -1, -1):
        #    rewards_l[i] = rewards_l[i] + agent.DISCOUNT * (1-dones_l[i]) * rewards_l[i+1]

        print("Frames: %d" % ((cycle + 1) * STEPS_BETWEEN_TRAINING))
        print("Param updates: %d" % (cycle * PARAM_UPDATES_PER_CYCLE))

        with tf.device('/device:CPU:0'):
          def dataGen():
            # TODO make sure we handle dones correctly in replay
            # before modification, ignored last entry of below (return[-1])
            return zip(zip(states_l, actions_l, rewards_l), dones_l)
          replay_buffer.addDatapoints(dataGen())

        def getBatch():
          with tf.device('/device:CPU:0'):
            # get rollout of length (INPUT_SHAPE[-1] + 1
            # this rollout is then stacked to get the transition from the
            # state represented by the first INPUT_SHAPE[-1] observations,
            # to the state represented by the last INPUT_SHAPE[-1] observations

            rollout,dones_ab,indices,probs = replay_buffer.sampleRolloutBatch(BATCH_SIZE,agent.INPUT_SHAPE[-1] + 1)
            dones_ab = tf.cast(dones_ab, tf.bool)
            state_roll, action_roll, rew_roll = rollout

            # get transition and action from rollouts
            states_a = state_roll[:,:-1,...]
            states_b = state_roll[:,1:,...]
            actions_a = action_roll[:,-2,...]
              
              
            # TODO should I use the k_probs?
            rollout_k,_,_,_ = replay_buffer.sampleRolloutBatch(BATCH_SIZE,agent.INPUT_SHAPE[-1])
            states_k = rollout_k[0]
            # move time dimension to last axis
            (states_a, states_b, states_k) = (np.moveaxis(a,1,-1) for a in (states_a, states_b, states_k))

            if agent.TOY_ENV:
              states_a, states_b, states_k = [s[...,-1:] for s in [states_a, states_b, states_k]] # remove time dimension
            #is this necessary?
            #states_a, states_b, states_k = [tf.reshape(x, [BATCH_SIZE] + agent.INPUT_SHAPE) for x in [states_a, states_b, states_k]]
            return (states_a, states_b, states_k, actions_a, dones_ab), indices, tf.cast(probs, tf.float32)

      else: # agent.TOY_ENV and not USE_BUFFER:

        def getBatch():
          states_a, states_b, states_k, actions, dones_ab = env.getRandomTransitions(BATCH_SIZE)
          indices = [0] * BATCH_SIZE; probs = [1.] * BATCH_SIZE # not important
          return (states_a, states_b, states_k, actions, dones_ab), indices, tf.cast(probs, tf.float32)
          
        
      
      start = timeit.default_timer()

      print('training distance')
      for b in range(PARAM_UPDATES_PER_CYCLE):
        #if b == 20:
        #  tf.profiler.experimental.start('logdir')
        #if b == 30:
        #  tf.profiler.experimental.stop()

        (states_a, states_b, states_k, actions_a, dones_ab), indices, probs = getBatch()

        with tf.GradientTape(watch_accessed_variables=True) as tape:
          loss, td_error, mean_dist = actor.loss(states_a, states_b, states_k, actions_a, probs, dones_ab)
          loss_TD = loss[0]; loss_ab = loss[1] 
          regloss = loss[2] * REGULARIZATION_WEIGHT
          

          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)
        grad = tape.gradient((loss_TD, loss_ab, regloss), actor.vars)
        #grad = [tf.clip_by_norm(g, GRAD_CLIP) for g in grad]

        # TODO: get rid of this is_nan check, slows down everything and it's stupid anyway.
        # need to figure out why these nans are happening.

        #for v in actor.vars:
        #  tf.debugging.check_numerics(v, 'nan before update')
        #foundNan = False
        #for g in grad:
        #  if tf.math.reduce_any(tf.math.is_nan(g)):
        #    foundNan = True
        #    nanCount += 1
        #    break
        #    #print(g)
        #    #tf.print(g, summarize=-1) # -1 indicates print everything
        #    #code.interact(local=locals())
        #  #tf.debugging.check_numerics(g, 'nan in gradients')
        #if foundNan:
        #  print('nanCount: ' + str(nanCount))
        #if not foundNan:
        agentOpt.apply_gradients(zip(grad, actor.vars))


        #for v in actor.vars:
        #  tf.debugging.check_numerics(v, 'nan after update')
        agent_losses += [loss]
        #printAccuracy(env, 100, actor)
        if agent.TOY_ENV and not b % 50:
          acc = actionAccuracy(env,100, actor)
          accs += [acc]
          #maxgrad = 0
          #for v in grad:
          #  maxgrad = max(tf.reduce_max(v), maxgrad)
          #max_grads.append(maxgrad.numpy())
          #maxweight = 0
          #for v in actor.vars:
          #  maxweight = max(tf.reduce_max(v), maxweight)
          #max_weights.append(maxweight.numpy())
        if not b % 50:
          print('Mean Dak: ' + str(mean_dist))
          print(loss_str)

        # important to convert td_error to numpy first,
        # otherwise it takes forever
        if USE_BUFFER:
          replay_buffer.updateWeights(indices, td_error.numpy())
        #maxtd = max(maxtd, tf.reduce_max(td_error))
        #print('MAX TD: ' + str(maxtd))



      end = timeit.default_timer()
      print("TIME: " + str(end - start))

      if USE_TARGET:
        copyWeightsFromTo(actor, target_actor)
         
    
      cycle += 1
      if not cycle % SAVE_CYCLES:
        print('Saving model...')
        tf.saved_model.save(actor, agent.model_savepath)
        with open(agent.picklepath, "wb") as fp:
          save['agent_losses'] = agent_losses
          save['accs'] = accs
          save['max_grads'] = max_grads
          save['max_weights'] = max_weights
          save['opt_config'] = agentOpt.get_config()
          save['opt_weights'] = agentOpt.get_weights()
          save['nanCount'] = nanCount
          pickle.dump(save, fp)

      
  
    
  
  
  
