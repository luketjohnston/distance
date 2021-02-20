import random

class PrioritizedReplayBuffer():
  def __init__(self, n, epsilon):
    assert(n & (n-1) == 0 and n != 0, 'n must be a power of 2')
    self.n = n
    self.epsilon = epsilon
    # invariant: sumtree[k] == sumtree[2*k+1] + sumtree[2*k+2]
    self.sumtree = [0 for _ in range(2 * n - 1)]
    self.data = [None for _ in range(n)]
    self.i = 0 # keeps track of where we are overwriting data in the buffer
    self.max_error = 1

    

  def sample(self):
    sumtree = self.sumtree
    sum_weights = sumtree[0]
  
    w = random.random() * sum_weights
    k = 0
    while k < self.n - 1:
      left,right = sumtree[2*k+1], sumtree[2*k+2]
      if w < left:
        k = 2*k+1
      else:
        k = 2*k+2
        w -= left
    # found leaf
    return self.data[k - self.n + 1], k 

  def sampleBatch(self, batch_size):
    batch = []
    indices = []
    for _ in range(batch_size):
      d,i = self.sample() 
      batch.append(d)
      indices.append(i)
    return batch, indices

  # indices are the indices into sumtree, as returned by sampleBatch
  def updateWeights(self, indices, weights):
     
    for i,index in enumerate(indices):
      self.sumtree[index] = weights[i] + self.epsilon
      self.max_error = max(weights[i], self.max_error)
    self.fixWeights(set(indices))

  # changed_indices are the indices into sutree
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

  # if error = -1, uses max error
  # dataGen is a generator
  def addDatapoints(self, dataGen, errors=None):
    
    if errors:
      self.max_error = max(max(errors), self.max_error)

    starti = self.i
    changed_indices = []
    for xi,x in enumerate(dataGen):
      self.data[self.i] = x
      error = self.max_error if not errors else errors[xi]
      self.sumtree[self.n - 1 + self.i] = error + self.epsilon
      changed_indices.append(self.n - 1 + self.i)
      self.i = (self.i + 1) % self.n
    self.fixWeights(changed_indices)

      
if __name__ == '__main__':
  buf =  PrioritizedReplayBuffer(16,0.001)
  buf.addDatapoints([-i for i in range(16)])
  for i in range(10):
    sample, indices = buf.sampleBatch(5)
    print('sample: ' + str(sample))
    print('indices: ' + str(indices))
    buf.updateWeights(indices, [i for _ in indices])
    input(buf.sumtree)
  buf.addDatapoints([i for i in range(8)])
  input('buf.data ' + str(buf.data))
  buf.addDatapoints([i for i in range(8)])
  input('buf.data ' + str(buf.data))
  input('buf.n' + str(buf.sumtree))

  
      
    

  
    
    


