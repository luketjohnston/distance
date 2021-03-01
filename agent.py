# TODO add optimizers to this model, so they can be saved and loaded easily
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import math
import os
from tensorflow.keras import Model
import gym
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from toyenv import ToyEnv, LoopEnv


# For some reason this is necessary to prevent error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ENVIRONMENT = 'MontezumaRevenge-v0'
ENVIRONMENT = 'PongDeterministic-v4'
#ENVIRONMENT = 'CartPole-v1'
env = gym.make(ENVIRONMENT)
TOYENV_SIZE = 10
USE_COORDS = False
env = ToyEnv(TOYENV_SIZE, USE_COORDS)

ALPHA = 2
BETA = 0


if USE_COORDS:
  INPUT_SHAPE = [2]
else:
  INPUT_SHAPE = [TOYENV_SIZE, TOYENV_SIZE, 1]



ACTIONS = env.action_space.n

ENT_EPSILON = 1e-7

HIDDEN_NEURONS=128

REGULARIZATION_WEIGHT = 0

FILTER_SIZES = [9, 5, 3]
CHANNELS =     [32,64,64]
STRIDES =     [4,2,1]

FILTER_SIZES = [7, 3]
CHANNELS =     [32,32]
STRIDES =     [4,1]
FILTER_SIZES = []
CHANNELS = []
STRIDES = []

ENCODING_SIZE = 32


IMSPEC = tf.TensorSpec([None] + INPUT_SHAPE,)
if ENVIRONMENT == 'CartPole-v1':
  IMSPEC = tf.TensorSpec([None, 4])

INTSPEC = tf.TensorSpec([None], dtype=tf.int64)
FLOATSPEC = tf.TensorSpec([None],)
DISTSPEC = tf.TensorSpec([None, ACTIONS],)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS],)
ENCSPEC = tf.TensorSpec([None, ENCODING_SIZE],)


DISCOUNT = 0.999
#DISCOUNT = 0.99
ENTROPY_WEIGHT = 0.001
EPSILON = 0.1

#ADD_ENTROPY = True

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'actor_save2')
accs_savepath = os.path.join(dir_path, 'accs.pickle')
loss_savepath = os.path.join(dir_path, 'actor_loss.pickle')
rewards_savepath = os.path.join(dir_path, 'rewards.pickle')



def getConvOutputSizeValid(w,h,filtersize, channels, stride):
  # padding if necessary
  w = (w - filtersize) // stride + 1
  h = (h - filtersize) // stride + 1
  return w,h,channels

  
  


class Agent(tf.Module):
  def __init__(self):
    super(Agent, self).__init__()
    self.vars = []

    size = INPUT_SHAPE
    if ENVIRONMENT == 'CartPole-v1':
      size = (4,1,1)
    for (f, c, s) in zip(FILTER_SIZES, CHANNELS, STRIDES):
      # first conv layer
      self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(f,f,size[2],c)), name='agent_conv'))
      size = getConvOutputSizeValid(size[0], size[1], f, c, s)

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(np.prod(size),HIDDEN_NEURONS)), name='Encoder_w'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Encoder_b'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ENCODING_SIZE)), name='Encoder_o'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,)), name='Encoder_bo'))

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE*2, HIDDEN_NEURONS)), name='Distance_w'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,)), name='Distance_b'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ACTIONS)), name='Distance_o'))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ACTIONS,)), name='Distance_bo'))


  ''' encodes state'''
  @tf.function(input_signature=(IMSPEC,))
  def encode(self, states):
    mvars = self.vars
    x = states
    #return x
    for i in range(len(CHANNELS)):
      filt = mvars[i]
      stride = STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,'VALID',name=None)
      x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Flatten()(x)
    vi = len(CHANNELS)
    #vi = 0
    w, b = mvars[vi:vi+2]
    encoding = tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[vi+2:vi+4]
    encoding = tf.nn.leaky_relu(tf.einsum('ba,ao->bo',encoding,w)  + b)
    return encoding

  ''' Takes two state encodings as input, and returns the distance between them for each action taken a'''
  @tf.function(input_signature=(ENCSPEC,ENCSPEC))
  def distance(self, enc1, enc2):
    # let's just start with something simple...
    mvars = self.vars[len(CHANNELS) + 4:]
    #mvars = self.vars[4:]
    x = tf.concat([enc1, enc2], axis=-1)
    w,b = mvars[:2]
    distance= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[2:4]
    distance = tf.einsum('ba,ao->bo',distance,w)  + b
    # TODO should this be RELU instead?
    #r = tf.stack([tf.reduce_sum(tf.abs(enc1 - enc2), -1) for _ in range(ACTIONS)], -1)
    return distance

  @tf.function(input_signature=(IMSPEC, IMSPEC))
  def distance_states(self, s1, s2):
    enc1 = self.encode(s1)
    enc2 = self.encode(s2)
    return self.distance(enc1, enc2)


  @tf.function(input_signature=(IMSPEC, IMSPEC, IMSPEC, INTSPEC, DISTSPEC, FLOATSPEC))
  def loss(self, states_a, states_b, states_k, action, Dbk_target, probs):
    enca, encb, enck = [self.encode(x) for x in [states_a, states_b, states_k]]
    
    Dak = self.distance(enca, enck)
    Dak_a = tf.gather(Dak, action, batch_dims=1)
    Dab = self.distance(enca, encb)
    Dab_a = tf.gather(Dab, action, batch_dims=1)

    # check if k == b. If so, target Dbk needs to be 0
    target = tf.reduce_min(Dbk_target, axis=-1)
    mask = tf.squeeze(tf.reduce_max(tf.cast(tf.not_equal(encb, enck), tf.float32), axis=1))
    target = target * mask
    target = tf.stop_gradient(1 + DISCOUNT * target)

    # apply importance sampling to the target
    weights = tf.pow(probs, -BETA)
    weights /= tf.reduce_max(weights)

    TD_error = tf.abs(Dak_a - target) 
    TD_error = tf.pow(TD_error, ALPHA)

    # TODO is this right?
    loss_TD = tf.reduce_mean(tf.pow(weights * (Dak_a - target), 2))
    #loss = tf.reduce_mean(tf.abs(Dak_a - target))
    # TODO having this update in here is going to mess things up when 
    # transitions are stochastic 
    loss_ab = tf.reduce_mean(tf.pow(weights * (Dab_a - 1), 2))
    #loss += tf.reduce_mean(tf.abs(Dab_a - 1))
    

    regloss = 0
    for x in self.vars:
      #regloss += REGULARIZATION_WEIGHT * tf.reduce_sum(tf.pow(x, 2))
      regloss += REGULARIZATION_WEIGHT * tf.reduce_sum(tf.abs(x))

    #tf.print('a,b,k, dak, dab, dbk_target')
    #tf.print(states_a)
    #tf.print(states_b)
    #tf.print(states_k)
    #tf.print(Dak)
    #tf.print(Dab)
    #tf.print(Dbk_target)

    return (loss_TD, loss_ab, regloss), TD_error
    
    
if __name__ == '__main__':

  #print('running loss')
  #encoder.loss_from_beginning(tf.zeros((16,84,110,1)))

  

  agent = Agent();

  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)


  losses = []
  episode_rewards = []
  accs = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
  with open(rewards_savepath, "wb") as fp:
    pickle.dump(episode_rewards, fp)
  with open(accs_savepath, "wb") as fp:
    pickle.dump(accs, fp)
    

    
    




    
