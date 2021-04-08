import gym
import code
import numpy as np
from gym import spaces

class ToyEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, n, use_coords=True):
    super(ToyEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(5)
    # Example for using image as input:
    if use_coords:
      self.observation_space = spaces.Box(low=0, high=n-1, shape=(2,), dtype=np.int32) 
    else:
      self.observation_space = spaces.Box(low=0, high=1, shape=(n,n), dtype=np.int32) 
    self.use_coords = use_coords
    self.coords = [0,0]
    self.n = n
  
  def getObs(self):
    if self.use_coords:
      obs = np.array(self.coords)
    else: 
      obs = np.zeros((self.n,self.n))
      obs[self.coords[0], self.coords[1]] = 1
    return obs

  def getRandomTransitions(self, batch_size):
    coords_a, coords_b, coords_k, actions, dones = [],[],[],[],[]
    for b in range(batch_size):
      a = np.random.randint(0,self.n,size=(2,))
      action = np.random.randint(0,5)
      b = a.copy()
      if action == 1 and a[0] < self.n-1: b[0] += 1
      if action == 2 and a[0] > 0:      b[0] -= 1
      if action == 3 and a[1] < self.n-1: b[1] += 1
      if action == 4 and a[1] > 0:      b[1] -= 1
      k = np.random.randint(0,self.n,size=(2,))
      coords_a.append(a)
      coords_b.append(b)
      coords_k.append(k)
      actions.append(action)
      dones.append(False)
    coords_a, coords_b, coords_k, actions, dones = [np.stack(c) for c in [coords_a, coords_b, coords_k, actions, dones]]
    obs_a, obs_b, obs_k = [self.coordsToObs(c) for c in [coords_a, coords_b, coords_k]]
    obs_a, obs_b, obs_k = [np.expand_dims(o, -1) for o in [obs_a, obs_b, obs_k]]
    return (obs_a, obs_b, obs_k, actions, dones)
      

  # coords has shape [batch size, 1, 1]
  def coordsToObs(self, coords):
    if self.use_coords:
      return np.array(coords)
    else:
      bs = coords.shape[0]
      obs = np.zeros((coords.shape[0], self.n, self.n))
      obs[range(bs), coords[:,0], coords[:,1]] = 1
      return obs


  def step(self, action):
    
    if not action in self.action_space: raise Exception
    if action == 1 and self.coords[0] < self.n - 1: self.coords[0] += 1
    if action == 2 and  self.coords[0] > 0: self.coords[0] -= 1
    if action == 3 and self.coords[1] < self.n - 1: self.coords[1] += 1
    if action == 4 and self.coords[1] > 0: self.coords[1] -= 1
    return self.getObs(), 0.0, False, {}

  def reset(self):
    self.coords = [0,0]
    return self.getObs()

  def render(self, mode='human', close=False):
    print(self.coords)

  def correctActions(self, start, dest):
    actions = []
    if start[0] < dest[0]:
      actions.append(1)
    if start[0] > dest[0]:
      actions.append(2)
    if start[1] < dest[1]:
      actions.append(3)
    if start[1] > dest[1]:
      actions.append(4)
    if np.array_equal(start, dest):
      actions.append(0)
      if start[0] == 0:
        actions.append(2)
      if start[1] == 0:
        actions.append(4)
      if start[0] == self.n - 1:
        actions.append(1)
      if start[1] == self.n - 1:
        actions.append(3)
    return actions


class LoopEnv(ToyEnv):
  def step(self, action):
    if not action in self.action_space: raise Exception
    if action == 1 and self.coords[0] < self.n - 1: self.coords[0] += 1
    if action == 2 and  self.coords[0] > 0: self.coords[0] -= 1
    if action == 3:
      if self.coords[1] < self.n - 1: self.coords[1] += 1
      else: self.coords[1] = 0
    if action == 4 and self.coords[1] > 0: self.coords[1] -= 1
    return self.getObs(), 0.0, False, {}

  def correctActions(self, start, dest):
    actions = []
    if start[0] < dest[0]:
      actions.append(1)
    if start[0] > dest[0]:
      actions.append(2)
    if start[1] < dest[1]:
      actions.append(3)
    if start[1] > dest[1]:
      if dest[1] + self.n - start[1] <= start[1] - dest[1]:
        actions.append(3)
      if dest[1] + self.n - start[1] >= start[1] - dest[1]:
        actions.append(4)
    if np.array_equal(start, dest):
      actions.append(0)
      if start[0] == 0:
        actions.append(2)
      if start[1] == 0:
        actions.append(4)
      if start[0] == self.n - 1:
        actions.append(1)
    return actions
  def getRandomTransitions(self, batch_size):
    raise Exception('not implemented')

class DeadEnd(ToyEnv):
  def step(self, action):
    self.coords[0] += 1
    done = False
    if self.coords[0] == self.n - 1:
      done = True
    return self.getObs(), 0.0, done, {}



if __name__ == '__main__':
  env = ToyEnv(5)
  #env.render()
  #env.step(0)
  #env.step(0)
  #env.step(1)
  #env.step(1)
  #env.step(2)
  #env.step(2)
  #env.step(3)
  #env.step(4)
  #env.render()
  code.interact(local=locals())


