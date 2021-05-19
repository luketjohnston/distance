import tensorflow as tf
import code
from replay import PrioritizedReplayBuffer
import multiprocessing as mp
import queue
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
from test import printAccs

LOAD_SAVE = True
PROFILE = False
CPU_CORES = 6

TERM_FLAG = 'terminate'

''' 
with alpha = 3 beta = 0, can use higher learning rate, and doesn't diverge (will temorarily diverge but always goes back)

'''

#TOY_ENV, 20x20
#ALPHA = 2
#ALPHA = 0

USE_BUFFER = False
TEST_LIMITED_STATES = False
BUFFER_SAMPLE_PROCESSES = 6


BUFFER_SIZE = 2**18 # 2**17 is max we can hold in memory it seems
BUFFER_SIZE = 2**10
#BUFFER_SIZE = 128

CLIP_REWARDS = True

USE_TARGET = False
SAVE_CYCLES = 1
BATCH_SIZE = 256 # use small batch size, otherwise sample_rollout dominates runtime.
BATCH_SIZE = 128
#BATCH_SIZE = 512
# TODO try to optimize replay_buffer, so can increase batch size. 
STEPS_BETWEEN_TRAINING = 512
#STEPS_BETWEEN_TRAINING = 64
PARAM_UPDATES_PER_CYCLE = 100 * BUFFER_SAMPLE_PROCESSES # should be multiple of BUFFER_SAMPLE_PROCESSES
#PARAM_UPDATES_PER_CYCLE = 50
SAVE_STATS_EVERY = 50

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
      dist_diffs = [] if not 'dist_diffs' in save else save['dist_diffs']
      SAVE_STATS_EVERY = SAVE_STATS_EVERY if not 'save_stats_every' in save else save['save_stats_every']
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

    # TODO: should I convert all these to numpy arrays? probably...
    states_l  = statelist[:-1]
    actions_l = [0 for _ in statelist[:-1]]
    rewards_l = [0 for _ in statelist[:-1]]
    dones_l   = [0 for _ in statelist[:-1]]

    if USE_BUFFER:
      data_type_and_shape = ((np.float32, saved_data_shape), (np.int32, ()), (np.float32, ()))
      replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, 0.001, data_type_and_shape)

      # TODO: add support for TOY_ENV and USE_BUFFER==False
      def getDataProcess(dataQueue, indicesQueue):
        for i in range(PARAM_UPDATES_PER_CYCLE // BUFFER_SAMPLE_PROCESSES):
          batch = []
          indices = indicesQueue.get()
          for (i1,i2) in indices:
            rollout, done_ab, index, prob = replay_buffer.getRolloutAt(i1, agent.INPUT_SHAPE[-1]+1)
            state_roll, action_roll, reward_roll = rollout
            # get transition and action from rollouts
            a = state_roll[:-1,...]
            b = state_roll[1:,...]
            action_a = action_roll[-2,...]
            # TODO should I use the k_probs?
            (k, _, _),_,_,_ = replay_buffer.getRolloutAt(i2, agent.INPUT_SHAPE[-1])
            # move time dimension to last axis
            # TODO is this slow? should I refactor agent.py so time axis is before channels axes?
            (states_a, states_b, states_k) = (np.moveaxis(a,0,-1) for a in (a, b, k))
            if agent.TOY_ENV:
              states_a, states_b, states_k = [s[...,-1:] for s in [states_a, states_b, states_k]] # remove time dimension
            prob = prob.astype(agent.NP_FLOAT_TYPE)
            action_a = action_a.astype(agent.NP_INT_TYPE)
            batch.append([states_a, states_b, states_k, action_a, done_ab, index, prob])
          a,b,k,act,done,ind,prob = [np.stack(x) for x in zip(*batch)]
          ind = ind.astype(agent.NP_INT_TYPE)
          dataQueue.put((a,b,k,act,done,ind,prob))

      def clearQueue(q):
        while True:
          try:
            q.get_nowait()
          except queue.Empty:
            return

      dataQueue = mp.Queue(maxsize = BUFFER_SAMPLE_PROCESSES * 4) # no particular reason for this size. Want to be able to buffer a few new datapoints if possible
      # needs to be big enough so it never blocks 
      indicesQueue = mp.Queue(maxsize = PARAM_UPDATES_PER_CYCLE)


      def putIndexBatchInQueue():
        batch = []
        for i in range(BATCH_SIZE):
          i1 = replay_buffer.sampleRolloutIndex(agent.INPUT_SHAPE[-1] + 1)
          i2 = replay_buffer.sampleRolloutIndex(agent.INPUT_SHAPE[-1])
          batch.append((i1,i2))
        try:
          indicesQueue.put_nowait(batch)
        except queue.Full:
          pass
      def datasetGenerator():
        while True:
          data  =  dataQueue.get()
          yield data

      output_sig = (
        agent.IMSPEC,
        agent.IMSPEC,
        agent.IMSPEC,
        agent.INTSPEC,
        agent.BOOLSPEC,
        agent.INTSPEC,
        agent.FLOATSPEC)

      dataset = tf.data.Dataset.from_generator(datasetGenerator, output_signature=output_sig)
      ###dataset = dataset.interleave(lambda x: makeDataset(), cycle_length = tf.data.AUTOTUNE, num_parallel_calls = tf.data.AUTOTUNE, deterministic=False) 

      dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
      dataiter = iter(dataset)

      replay_buffer.addDatapoints(zip(zip(states_l, actions_l, rewards_l), dones_l))

    elif agent.TOY_ENV and not USE_BUFFER:
      def datasetGenerator():
        while True:
          a, b, k, action, done_ab = env.getRandomTransitions(BATCH_SIZE)
          index = [0] * BATCH_SIZE; prob = [1.] * BATCH_SIZE # indices and probs not important when not using buffer
          yield (a, b, k, action, done_ab, index, prob)
      dataiter = iter(datasetGenerator())
    else:
      raise Exception('cant have USE_BUFFER==false if not using environment from toyenv.py')



    while True: 

      if cycle == 1 and PROFILE:
        tf.profiler.experimental.start('logdir')
      if cycle == 2 and PROFILE:
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
          print('done adding to replay_buffer')


      try: # need to catch KeyboardInterrupt, to handle terminating processes

        #index_proc = mp.Process(target=indexProcess, args=(indicesQueue, updateQueue))
        ##index_proc.daemon = True
        #index_proc.start()

        # have to start processes here (can't have them running when replay_buffer is being updated
        # with new datapoints)
        if USE_BUFFER:
          sample_procs = []
          for p in range(BUFFER_SAMPLE_PROCESSES):
            sample_procs.append(mp.Process(target = getDataProcess, args=(dataQueue, indicesQueue)))
            sample_procs[-1].start()

          for p in sample_procs:
            putIndexBatchInQueue()
        
        start = timeit.default_timer()

        print('training distance')
        for b in range(PARAM_UPDATES_PER_CYCLE):
          (states_a, states_b, states_k, actions_a, dones_ab, indices, probs) = next(dataiter)

          loss, td_error, mean_dist = actor.loss(states_a, states_b, states_k, actions_a, probs, dones_ab)

          (loss_TD, loss_ab, regloss) = loss

          loss_str = ''.join('{:6f}, '.format(lossv) for lossv in loss)

          agent_losses += [loss]
          #printAccuracy(env, 100, actor)
          if agent.TOY_ENV and not b % SAVE_STATS_EVERY:
            acc,dist_diff = printAccs(env,100, actor, agent.MAX_DIST, logDist=agent.USE_LOG)
            accs += [acc]
            dist_diffs += [dist_diff]
          if not b % SAVE_STATS_EVERY:
            print('Mean Dak: ' + str(mean_dist))
            print(loss_str)

          # important to convert td_error to numpy first,
          # otherwise it takes forever
          if USE_BUFFER:
            # TODO is it possibly to pass tensor to process (without converting to numpy?) 
            #updateQueue.put((indices.numpy(), td_error.numpy()))
            replay_buffer.updateWeights(indices.numpy(), td_error.numpy())
            putIndexBatchInQueue()

          #maxtd = max(maxtd, tf.reduce_max(td_error))
          #print('MAX TD: ' + str(maxtd))

          if b == PARAM_UPDATES_PER_CYCLE - 1: break


        if USE_BUFFER:
          for p in sample_procs:
            p.join()
          clearQueue(indicesQueue)

      except KeyboardInterrupt:
        if USE_BUFFER:
          for p in sample_procs:
            p.terminate()
        raise KeyboardInterrupt

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
          save['dist_diffs'] = dist_diffs
          save['max_grads'] = max_grads
          save['max_weights'] = max_weights
          save['save_stats_every'] = SAVE_STATS_EVERY
          save['nanCount'] = nanCount
          pickle.dump(save, fp)

      
  
    
  
  
  
