import gym
import code
import numpy as np
from gym import spaces

class ToyEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, n):
    super(ToyEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(5)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=n-1, shape=(2,), dtype=np.int32) 
    self.coords = [0,0]
    self.n = n
  
  def getObs(self):
    #obs = np.zeros((self.n,self.n))
    #obs[self.coords] = 1
    return np.array(self.coords)


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


