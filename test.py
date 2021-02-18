
import numpy as np
import tensorflow as tf
import agent

def printAccuracy(batch_size, actor):
  coords = np.random.randint(0,agent.WIDTH,size=(batch_size, 2, 2, 1))
  #obs = np.zeros((batch_size, 2, agent.WIDTH, agent.HEIGHT))
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
  print('accurracy: ' + str(tf.reduce_mean(tf.abs(dists - truths))))

def actionAccuracy(batch_size, actor):
  coords = np.random.randint(0,agent.WIDTH,size=(batch_size, 2, 2, 1))
  # TODO This isn't eactly correct, distance from state to same state will be 1, not 0
  enc1 = actor.encode(coords[:,0,:])
  enc2 = actor.encode(coords[:,1,:])
  dists = actor.distance(enc1, enc2)
  actions = tf.math.argmin(dists, axis=-1).numpy()
  correct = 0
  for i in range(batch_size):
    action = actions[i]
    if action == 0:
      correct += int(np.array_equal(coords[i,1], coords[i,0]))
    if action == 1:
      correct += int(coords[i,1,0] > coords[i,0,0] or (np.array_equal(coords[i,1], coords[i,0]) and coords[i,1,0] == (agent.WIDTH - 1)))
    if action == 2:
      correct += int(coords[i,1,0] < coords[i,0,0] or (np.array_equal(coords[i,1], coords[i,0]) and coords[i,1,0] == 0))
    if action == 3:
      correct += int(coords[i,1,1] > coords[i,0,1] or (np.array_equal(coords[i,1], coords[i,0]) and coords[i,1,1] == (agent.WIDTH-1)))
    if action == 4:
      correct += int(coords[i,1,1] < coords[i,0,1] or (np.array_equal(coords[i,1], coords[i,0]) and coords[i,1,1] == 0))
  print('action acc: ' + str(correct / batch_size))

def printRandomDist(actor):
  coords = np.random.randint(0,agent.WIDTH,size=(1, 2, 2, 1))
  truths = np.sum(np.abs(coords[:,0,:] - coords[:,1,:]), axis=-2)
  print(coords)
  print(truths)
  dist = printDist(coords[:,0,:], coords[:,1,:], actor)
  return coords, dist

def printLargestDist(actor):
  # TODO why is the 1 dim at the end necessary?
  coord1 = np.array([[[0],[0]]])
  coord2 = np.array([[[agent.WIDTH-1],[agent.WIDTH-1]]])
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
  coords = np.random.randint(0,agent.WIDTH,size=(1, 2, 2, 1))
  coords[:,0,:] = coords[:,1,:]
  return coords, printDist(coords[:,0,:], coords[:,1,:], actor)

if __name__ == '__main__':
  while True:
    actor = tf.saved_model.load(agent.model_savepath)
    printLargestDist(actor)
    input('above is largest')
    printSelfDist(actor)
    input('above is self dist')
    correct = 0
    coords, dists = printRandomDist(actor)
    action = tf.math.argmin(dists, axis=-1).numpy()
    if action == 0:
      correct += int(np.array_equal(coords[0,1], coords[0,0]))
    if action == 1:
      correct += int(coords[0,1,0] > coords[0,0,0] or (np.array_equal(coords[0,1], coords[0,0]) and coords[0,1,0] == (agent.WIDTH - 1)))
    if action == 2:
      correct += int(coords[0,1,0] < coords[0,0,0] or (np.array_equal(coords[0,1], coords[0,0]) and coords[0,1,0] == 0))
    if action == 3:
      correct += int(coords[0,1,1] > coords[0,0,1] or (np.array_equal(coords[0,1], coords[0,0]) and coords[0,1,1] == (agent.WIDTH-1)))
    if action == 4:
        correct += int(coords[0,1,1] < coords[0,0,1] or (np.array_equal(coords[0,1], coords[0,0]) and coords[0,1,1] == 0))
    if not correct: input("ACTION WRONG")
