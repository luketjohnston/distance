
import code
import numpy as np
import tensorflow as tf
import agent
from toyenv import LoopEnv

def printAccuracy(batch_size, actor):
  coords = np.random.randint(0,agent.TOYENV_SIZE,size=(batch_size, 2, 2, 1))
  #obs = np.zeros((batch_size, 2, agent.TOYENV_SIZE, agent.HEIGHT))
  #obs[np.arange(batch_size), 0, coords[:,0,0], coords[:,0,1]] = 1
  #obs[np.arange(batch_size), 1, coords[:,1,0], coords[:,1,1]] = 1
  # TODO This isn't eactly correct, distance from state to same state will be 1, not 0
  truths = np.sum(np.abs(coords[:,0,:] - coords[:,1,:]), axis=-2)
  #obs = tf.cast(obs, tf.float32)
  #obs = tf.expand_dims(obs, -1)
  #enc1 = actor.encode(obs[:,0,:,:,:])
  #enc2 = actor.encode(obs[:,1,:,:,:])
  enc1 = actor.encode(coords[:,0,:])
  enc2 = actor.encode(coords[:,1,:])
  dists = actor.distance(enc1, enc2)
  dists = tf.reduce_min(dists, axis=-1)
  truths = tf.cast(tf.squeeze(truths), tf.float32)
  ave_err = tf.reduce_mean(tf.abs(dists - truths))
  print('accurracy: ' + str(ave_err))
  return ave_err

def actionAccuracy(env, batch_size, actor):
  coords = np.random.randint(0,agent.TOYENV_SIZE,size=(batch_size, 2, 2, 1))
  enc1 = actor.encode(coords[:,0,:])
  enc2 = actor.encode(coords[:,1,:])
  dists = actor.distance(enc1, enc2)
  actions = tf.math.argmin(dists, axis=-1).numpy()
  correct = 0
  for i in range(batch_size):
    action = actions[i]
    correct += int(action in env.correctActions(coords[i,0], coords[i,1]))
  acc = correct / batch_size
  print('action acc: ' + str(acc))
  return acc

def printRandomDist(actor):
  coords = np.random.randint(0,agent.TOYENV_SIZE,size=(1, 2, 2, 1))
  truths = np.sum(np.abs(coords[:,0,:] - coords[:,1,:]), axis=-2)
  print(coords)
  print(truths)
  dist = printDist(coords[:,0,:], coords[:,1,:], actor)
  return coords, dist

def printLargestDist(actor):
  # TODO why is the 1 dim at the end necessary?
  coord1 = np.array([[[0],[0]]])
  coord2 = np.array([[[agent.TOYENV_SIZE-1],[agent.TOYENV_SIZE-1]]])
  print('Dist between %s and %s: ' % (str(coord1), str(coord2)))
  dist = printDist(coord1, coord2, actor)
  return dist

def printDist(c1, c2, actor):
  enc1 = actor.encode(c1)
  enc2 = actor.encode(c2)
  dist = actor.distance(enc1, enc2)
  print(dist)
  return dist

def printSelfDist(actor):
  coords = np.random.randint(0,agent.TOYENV_SIZE,size=(1, 2, 2, 1))
  coords[:,0,:] = coords[:,1,:]
  return coords, printDist(coords[:,0,:], coords[:,1,:], actor)

if __name__ == '__main__':
  actor = tf.saved_model.load(agent.model_savepath)
  env = LoopEnv(agent.TOYENV_SIZE)
  printLargestDist(actor)
  input('above is largest')
  printSelfDist(actor)
  input('above is self dist')
  #while True:

  #  coords = np.random.randint(0,agent.TOYENV_SIZE,size=(128, 2, 2, 1))
  #  enc1 = actor.encode(coords[:,0,:])
  #  enc2 = actor.encode(coords[:,1,:])
  #  dists = actor.distance(enc1, enc2)
  #  actions = tf.math.argmin(dists, axis=-1).numpy()
  #  correct = 0
  #  for i in range(128):
  #    action = actions[i]
  #    if not action in env.correctActions(coords[i,0], coords[i,1]):
  #      print('incorrect action: ' + str(action))
  #      print(coords[i,:,:])
  #      input(dists[i,:])
  #    else:     
  #      print('correct...')
  

  ul = [[[0],[0]]]
  ur = [[[0],[agent.TOYENV_SIZE -1]]]
  bl = [[[agent.TOYENV_SIZE-1],[0]]]
  br = [[[agent.TOYENV_SIZE-1],[agent.TOYENV_SIZE-1]]]

  states = [[None for _ in range(agent.TOYENV_SIZE)] for _ in range(agent.TOYENV_SIZE)]
  for i in range(agent.TOYENV_SIZE):
    for j in range(agent.TOYENV_SIZE):
      states[i][j] = [[[i],[j]]]
  def dist(t1,t2):
    i1, j1 = t1
    i2,j2 = t2
    return actor.distance_states(states[i1][j1], states[i2][j2])

  code.interact(local=locals())