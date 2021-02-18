import gym
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

  def check_accurracy(self, batch_size):
    coords = np.random.randint(0,self.n,size=(batch_size, 2, 2))
    #observations = np.zeros((batch_size, 2, self.n, self.n))
    #observations[np.arange(batch_size), 0, coords[:,0,0], coords[:,0,1]] = 1
    #observations[np.arange(batch_size), 1, coords[:,1,0], coords[:,1,1]] = 1
    truth1 = np.sum(np.abs(coords[:,0,:] - coords[:,1,:]), axis=-1)
    #print(observations[0,0,:,:])
    #print(observations[0,1,:,:])
    print(truth1)



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

  env.check_accurracy(3)

