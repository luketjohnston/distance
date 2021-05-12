import random
from utility import loopListBefore
import numpy as np


class PrioritizedReplayBuffer():
  ''' 
  n: size of buffer. must be a power of 2
  epsilon: added to every priority entry in the replay buffer
  data_types:  a list of (t, s) tuples, where for each
    tuple, we need to store data of type t and shape s '''
  def __init__(self, n, epsilon, data_types):
    assert n & (n-1) == 0 and n != 0, 'n must be a power of 2'
    self.n = n
    self.epsilon = epsilon
    # invariant: sumtree[k] == sumtree[2*k+1] + sumtree[2*k+2]
    self.sumtree = np.zeros(shape=(2*n-1,))
    self.data = [np.empty(dtype=dt, shape=((n,) + s)) for (dt,s) in data_types]
    self.datacount = 0
    self.dones = np.array([False for _ in range(n)])
    self.i = 0 # keeps track of where we are overwriting data in the buffer
    self.max_error = 1
    

  def sampleSumtreeIndex(self):
    sumtree = self.sumtree
    sum_weights = sumtree[0]
    w = random.random() * sum_weights
    k = 0
    while k < self.n - 1:
      left,right = sumtree[2*k+1], sumtree[2*k+2]
      if w <= left:
        k = 2*k+1
      else:
        k = 2*k+2
        w -= left
    # found leaf
    return k

  def uniformSampleState(self):
    k = random.randrange(self.datacount)
    i = k - self.n + 1
    return tuple((d[i] for d in self.data))
    

  # 'rollout' is the number of states in the rollout
  def sampleRollout(self, rollout):
    while True:
      k = self.sampleSumtreeIndex()
      i = k - self.n + 1
      terminal_i = self.i
      if terminal_i < i: terminal_i += self.n
      # if the rollout would overlap the edge of the memory buffer 
      # (self.i records slot for next input data), or if the rollout
      # has a "done" that's not at the end, it is invalid and we need
      # to try again.
      if (not terminal_i in range(i+1,i+rollout)) and not True in np.take(self.dones, range(i,i+rollout-1),mode='wrap'):
        break
    data_r = tuple((np.take(d,range(i,i+rollout),axis=0,mode='wrap') for d in self.data))
    return data_r, self.dones[i % self.n], k, self.sumtree[k]

  def uniformBatchState(self, batch_size):
    data_batches = tuple(([] for _ in self.data))
    for _ in range(batch_size):
      state = self.uniformSampleState()
      for j,d in enumerate(state):
        data_batches[j].append(d)
    data_batches = tuple((np.stack(d) for d in data_batches))
    return data_batches

  ''' return values:
  batch: batch[i] is a list of subsequent entries of self.data
    (a rollout). No entry can have self.dones[i] set except the last
  dones: dones[i] is the self.dones value for the last entry of batch[i]
  indices: the indices of the last states of each rollout. Used
    in a later call to updateWeights
  probs: the prob of returning each returned rollout
  '''
  def sampleRolloutBatch(self, batch_size, rollout):
    data_batches = tuple(([] for _ in self.data))
    probs = []
    dones = []
    indices = []
    for _ in range(batch_size):
      rollouts,done,i,p = self.sampleRollout(rollout) 
      for j,d in enumerate(rollouts):
        data_batches[j].append(d)
      dones.append(done)
      indices.append(i)
      probs.append(p)
    data_batches = tuple((np.stack(d) for d in data_batches))
    return data_batches, dones, indices, probs

  # indices are the indices into sumtree, as returned by sampleBatch
  def updateWeights(self, indices, weights):
    for i,index in enumerate(indices):
      self.sumtree[index] = weights[i] + self.epsilon
      self.max_error = max(weights[i], self.max_error)
    self.fixWeights(set(indices))

  # changed_indices are the indices into subtree
  def fixWeights(self, changed_indices):
    # percolate changes to sums up tree
    while not 0 in changed_indices:
      parents = set()
      for k in changed_indices:
        p = (k-1) // 2
        parents.add(p)
        left, right = self.sumtree[2*p+1], self.sumtree[2*p+2]
        self.sumtree[p] = left + right
      changed_indices = parents

  ''' add many datapoints at once. Slightly faster than adding
  one datapoint at a time, because we can update all the weights of
  the newly added datapoints at once, after adding all of them
  (instead of updating individually every step)

  if error = -1, uses max error
  dataGen is a generator, generates (x,done) where x is data, done is 
    the done value of that data
  '''
  def addDatapoints(self, dataGen, errors=None):
    if errors:
      self.max_error = max(max(errors), self.max_error)
    starti = self.i
    changed_indices = []

    for xi,(data,done) in enumerate(dataGen):
      if self.datacount < self.n:
        self.datacount += 1
      for j,d in enumerate(data):
        self.data[j][self.i] = d
      self.dones[self.i] = done
      error = self.max_error if not errors else errors[xi]
      self.sumtree[self.n - 1 + self.i] = error + self.epsilon
      changed_indices.append(self.n - 1 + self.i)
      self.i = (self.i + 1) % self.n
    self.fixWeights(changed_indices)

  ''' to add one datapoint at a time, use this. While saving up all datapoints
  from an experience and then adding at once is more efficient, if we are
  trying to maximize experience replay buffer size, we want to add them
  instantly (so we don't have to store them in memory twice)
   
  TODO I've never actually tested this, wrote it and then decided not to use
  it.
  '''
  def addDatapoint(self, dp, done, error=None):
    if errors:
      self.max_error = max(error, self.max_error)
    if self.datacount < self.n:
      self.datacount += 1
    for i,d in enumerate(data):
      self.data[i][self.i] = d
    self.dones[self.i] = done
    error = self.max_error if not error else error
    self.sumtree[self.n - 1 + self.i] = error + self.epsilon
    self.fixWeights([self.n - 1 + self.i])
 


      
if __name__ == '__main__':
  buf =  PrioritizedReplayBuffer(16,0.001, ((np.int32, ()),))
  buf.addDatapoints([((-i,),False) for i in range(24)])
  for i in range(10):
    sample, dones, indices, probs = buf.sampleRolloutBatch(20,5)
    print('sample: ' + str(sample))
    print('indices: ' + str(indices))
    buf.updateWeights(indices, [i for _ in indices])
    input(buf.sumtree)
  buf.addDatapoints([((i,),False) for i in range(8)])
  input('buf.data ' + str(buf.data))
  buf.addDatapoints([((i,),False) for i in range(8)])
  input('buf.data ' + str(buf.data))
  input('buf.n' + str(buf.sumtree))

  
      
    

  
    
    


