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

''' 
with alpha = 3 beta = 0, can use higher learning rate, and doesn't diverge (will temorarily diverge but always goes back)

'''

#TOYENV, 20x20
#ALPHA = 2
#ALPHA = 0

OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'Adam' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER_EPSILON = 1e-7 #1e-7 is default for adam
GRAD_CLIP = 0.01
REGULARIZATION_WEIGHT = 0.0

BUFFER_SIZE = 2**15
#BUFFER_SIZE = 8


CLIP_REWARDS = True
#LR = 0.000001 #use this for fully connected
#LR = 0.001 # this is tf default
LR = 0.000003 # this is the best so far for toyenv, diverges at an order of magnitude higher.
LR = 0.00003 # diverges here! UNLESS we use experience replay and then it seems to work ok
LR = 0.01 

USE_TARGET = False
SAVE_CYCLES = 1
BATCH_SIZE = 128
ENVS = 2
STEPS_BETWEEN_TRAINING = 512
PARAM_UPDATES_PER_CYCLE = 500
#TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP  = 20

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
    envs = []
    states = []
    statelists = []

    # TODO: implement tf.data for optimization
    if not agent.TOY_ENV:
      def preprocess(s):
        s = tf.image.rgb_to_grayscale(s)
        s = tf.image.resize(s,(agent.INPUT_SHAPE[0],agent.INPUT_SHAPE[1]))
        return tf.cast(s / 255.0, tf.float32)
    else:
      def preprocess(s):
        return tf.cast(s, tf.float32)

    # make environment
    for i in range(ENVS):
 
      env = agent.makeEnv()
      state1 = tf.cast(env.reset(), tf.float32)
      # 0 action is 'NOOP'
      state2 = tf.cast(env.step(0)[0], tf.float32)
      state3 = tf.cast(env.step(0)[0], tf.float32)
      state4 = tf.cast(env.step(0)[0], tf.float32)
      statelist = [state1, state2, state3, state4]
      statelist = [preprocess(s) for s in statelist]
      state = tf.stack(statelist, -1)
      envs += [env]
      states.append(state)
      statelists += [statelist]

    states = tf.stack(states)

    cycle = 0
    nanCount = 0 if not 'nanCount' in save else save['nanCount']
    total_rewards = [0 for _ in envs]

    states_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, 0.001)
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, 0.001)

    while True: 


      if not agent.TOY_ENV:
        states_l, actions_l, rewards_l, dones_l = [],[],[],[]
        print('acting')

        if cycle == 0:
          cycle_steps = BUFFER_SIZE // ENVS
        else:
          cycle_steps = STEPS_BETWEEN_TRAINING
        for step in range(cycle_steps):
          actions = np.random.randint(0,5,size=(ENVS,))
          next_states, rewards, dones = [], [], []

          for i in range(ENVS):
            observation, reward, done, info = envs[i].step(actions[i])
            if CLIP_REWARDS:
              if reward > 1: reward = 1.0
              if reward < -1: reward = -1.0
            total_rewards[i] += reward
            observation = preprocess(observation)
            if (done): 
              envs[i].reset()
              episode_rewards += [total_rewards[i]]
              print("Finished episode %d, reward: %f" % (len(episode_rewards), total_rewards[i]))
              total_rewards[i] = 0

            statelists[i] = statelists[i][1:]
            statelists[i].append(observation)
            next_state = tf.stack(statelists[i], -1)
            next_states.append(next_state)
            dones.append(float(done))
            rewards.append(reward)

          # need to copy to CPU so we don't use all the GPU memory
          with tf.device('/device:CPU:0'):
          #if True:
            states_l.append(tf.identity(states))
            actions_l.append(tf.identity(actions))
            rewards_l.append(tf.stack(rewards))
            dones_l.append(tf.stack(dones))

          states = tf.stack(next_states)


        with tf.device('/device:CPU:0'):
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

        with tf.device('/device:CPU:0'):
          def dataGen():
            for i in range(len(states_l) - 1):
              envs_data = (states_l[i], states_l[i+1], actions_l[i], rewards_l[i], dones_l[i])
              for e,_ in enumerate(envs):
                # TODO this is ok right? only add transitions that don't end the episode?
                if not dones_l[i][e]:
                  dp = [ed[e] for ed in envs_data]
                  yield dp
                

          #replay_buffer.addDatapoints(dataGen(), [1 for _ in range((len(states_l) - 1) * TRANSITION_GOAL_PAIRS_ADDED_PER_TIMESTEP * ENVS)])
          replay_buffer.addDatapoints(dataGen())

          for e in range(ENVS):
            states_buffer.addDatapoints([s[e] for s in states_l])

        def getBatch():
          with tf.device('/device:CPU:0'):
            # TODO: there's an error here, doesn't take into account dones. Shouldn't matter for toy environment though.
            b,indices,probs = replay_buffer.sampleBatch(BATCH_SIZE)
            states_a, states_b, actions_a, _, _ = zip(*b)
            # TODO should I use the k_probs?
            states_k, k_indices, k_probs = states_buffer.sampleBatch(BATCH_SIZE)
            states_a, states_b, states_k, actions_a = [tf.stack(s) for s in [states_a, states_b, states_k, actions_a]]
            if agent.TOY_ENV:
              states_a, states_b, states_k = [s[...,-1:] for s in [states_a, states_b, states_k]] # remove time dimension
            #is this necessary?
            states_a, states_b, states_k = [tf.reshape(x, [BATCH_SIZE] + agent.INPUT_SHAPE) for x in [states_a, states_b, states_k]]
            Dbk_target = target_actor.distance_states(states_b, states_k)
            return (states_a, states_b, states_k, actions_a, Dbk_target), indices, tf.cast(probs, tf.float32)

      else: # agent.TOYENV:

        def getBatch():
          states_a, states_b, states_k, actions = env.getRandomTransitions(BATCH_SIZE)
          Dbk_target = target_actor.distance_states(states_b, states_k)
          indices = [0] * BATCH_SIZE; probs = [1.] * BATCH_SIZE # not important
          return (states_a, states_b, states_k, actions, Dbk_target), indices, tf.cast(probs, tf.float32)
          
        
      
      start = timeit.default_timer()

      print('training distance')
      for b in range(PARAM_UPDATES_PER_CYCLE):
        (states_a, states_b, states_k, actions_a, Dbk_target), indices, probs = getBatch()
        with tf.GradientTape(watch_accessed_variables=True) as tape:
          loss, td_error = actor.loss(states_a, states_b, states_k, actions_a, Dbk_target, probs)
          loss_TD = loss[0]; loss_ab = loss[1] 
          regloss = loss[2] * REGULARIZATION_WEIGHT
          

          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)
        grad = tape.gradient((loss_TD, loss_ab, regloss), actor.vars)
        #grad = [tf.clip_by_norm(g, GRAD_CLIP) for g in grad]

        for v in actor.vars:
          tf.debugging.check_numerics(v, 'nan before update')
        foundNan = False
        for g in grad:
          if tf.math.reduce_any(tf.math.is_nan(g)):
            foundNan = True
            nanCount += 1
            #print(g)
            #tf.print(g, summarize=-1) # -1 indicates print everything
            #code.interact(local=locals())
          #tf.debugging.check_numerics(g, 'nan in gradients')
        if foundNan:
          print('nanCount: ' + str(nanCount))
        if not foundNan:
          agentOpt.apply_gradients(zip(grad, actor.vars))
        for v in actor.vars:
          tf.debugging.check_numerics(v, 'nan after update')
        agent_losses += [loss]
        #printAccuracy(env, 100, actor)
        if agent.TOY_ENV and not b % 50:
          acc = actionAccuracy(env,100, actor)
          accs += [acc]
          print(loss_str)
          maxgrad = 0
          for v in grad:
            maxgrad = max(tf.reduce_max(v), maxgrad)
          max_grads.append(maxgrad.numpy())
          maxweight = 0
          for v in actor.vars:
            maxweight = max(tf.reduce_max(v), maxweight)
          max_weights.append(maxweight.numpy())

        # important to convert td_error to numpy first,
        # ovtherwise it takes forever
        if not agent.TOY_ENV:
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

      
  
    
  
  
  
