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
from toyenv import ToyEnv, LoopEnv, DeadEnd


# For some reason this is necessary to prevent error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

USE_LOG = False

TOY_ENV = True
TOYENV_SIZE = 20
USE_COORDS = True
DEADEND = False


AB_WEIGHT = 1.0
REGULARIZATION_WEIGHT = 0.0

ENVIRONMENT = 'MontezumaRevengeDeterministic-v4'
#ENVIRONMENT = 'PongDeterministic-v4'
#ENVIRONMENT = 'CartPole-v1'

#OPTIMIZER = 'SGD' # 'Adam' or 'RMSprop' or 'SGD'
#OPTIMIZER = 'RMSprop' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER = 'Adam' # 'Adam' or 'RMSprop' or 'SGD'
OPTIMIZER_EPSILON = 1e-7 #1e-7 is default for adam
GRAD_CLIP = 0.01

#LR = 0.01 
#LR = 0.001 # this is tf default
#LR = 0.0001  # this seems to be the best for 20x20 toyenv, usecoords=True, uselog=False
LR = 0.00001  # works for 20x20 toyenv
#LR = 0.000001  # too low for 20x20 toyenv, log

def makeEnv():
  if TOY_ENV and not DEADEND:
    return ToyEnv(TOYENV_SIZE, USE_COORDS)
  elif TOY_ENV and DEADEND:
    return DeadEnd(TOYENV_SIZE, USE_COORDS)
  else:
    return gym.make(ENVIRONMENT)


env = makeEnv()
ACTIONS = env.action_space.n

if TOY_ENV:
  if USE_COORDS:
    INPUT_SHAPE = [2, 1]
  else:
    INPUT_SHAPE = [TOYENV_SIZE, TOYENV_SIZE, 1]
  ACTION_MAP = [0,1,2,3,4]
else:
  INPUT_SHAPE = [84,110,4]
  ACTIONS = 8
  ACTION_MAP = [0,1,2,3,4,5,11,12]

ALPHA = 0
BETA = 0



ENT_EPSILON = 1e-7

HIDDEN_NEURONS=128

FILTER_SIZES = [7, 5, 5]
CHANNELS =     [32,64,64]
STRIDES =     [2,2,1]

#FILTER_SIZES = [7, 3]
#CHANNELS =     [32,32]
#STRIDES =     [4,1]

if TOY_ENV:
  FILTER_SIZES = []
  CHANNELS = []
  STRIDES = []

ENCODING_SIZE = 64


# TODO can't change to float16, dunno why
FLOAT_TYPE = tf.float32
NP_FLOAT_TYPE = np.float32
NP_INT_TYPE = np.int32
INT_TYPE = tf.int32
IMSPEC = tf.TensorSpec([None] + INPUT_SHAPE, dtype=FLOAT_TYPE)
#if ENVIRONMENT == 'CartPole-v1':
#  IMSPEC = tf.TensorSpec([None, 4])

INTSPEC = tf.TensorSpec([None], dtype=INT_TYPE)
FLOATSPEC = tf.TensorSpec([None], dtype=FLOAT_TYPE)
DISTSPEC = tf.TensorSpec([None, ACTIONS], dtype=FLOAT_TYPE)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS], dtype=FLOAT_TYPE)
ENCSPEC = tf.TensorSpec([None, ENCODING_SIZE], dtype=FLOAT_TYPE)


DISCOUNT = 0.999

if USE_LOG:
  MAX_DIST = np.exp(10.0, dtype=np.float32)
else:
  MAX_DIST = 1 / (1 - DISCOUNT)

#ADD_ENTROPY = True

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'actor_save')
picklepath = os.path.join(model_savepath, 'actor.pickle')

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

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(np.prod(size),HIDDEN_NEURONS),dtype=FLOAT_TYPE), name='Encoder_w', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,),dtype=FLOAT_TYPE), name='Encoder_b', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ENCODING_SIZE),dtype=FLOAT_TYPE), name='Encoder_o', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE,),dtype=FLOAT_TYPE), name='Encoder_bo', dtype=FLOAT_TYPE))

    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ENCODING_SIZE*2, HIDDEN_NEURONS),dtype=FLOAT_TYPE), name='Distance_w', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,),dtype=FLOAT_TYPE), name='Distance_b', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(HIDDEN_NEURONS,ACTIONS),dtype=FLOAT_TYPE), name='Distance_o', dtype=FLOAT_TYPE))
    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(ACTIONS,),dtype=FLOAT_TYPE), name='Distance_bo', dtype=FLOAT_TYPE))


    opt_params = {'learning_rate': LR, 'clipvalue': GRAD_CLIP}
    if OPTIMIZER == 'Adam':
      opt = tf.keras.optimizers.Adam
      opt_params['epsilon'] = OPTIMIZER_EPSILON
    elif OPTIMIZER == 'RMSprop':
      opt = tf.keras.optimizers.RMSprop
      opt_params['epsilon'] = OPTIMIZER_EPSILON
    elif OPTIMIZER == 'SGD':
      opt = tf.keras.optimizers.SGD
    self.opt = opt(**opt_params)


  ''' encodes state'''
  @tf.function(input_signature=(IMSPEC,))
  def encode(self, states):
    #tf.debugging.check_numerics(states, 'states nan')
    mvars = self.vars
    x = states
    for i in range(len(CHANNELS)):
      filt = mvars[i]
      stride = STRIDES[i]
      x = tf.nn.conv2d(x,filt,stride,'VALID',name=None)
      x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Flatten()(x)
    vi = len(CHANNELS)
    w, b = mvars[vi:vi+2]
    encoding = tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[vi+2:vi+4]
    encoding = tf.nn.leaky_relu(tf.einsum('ba,ao->bo',encoding,w)  + b)
    return encoding

  ''' Takes two state encodings as input, and returns the distance between them for each action option a'''
  @tf.function(input_signature=(ENCSPEC,ENCSPEC))
  def distance(self, enc1, enc2):
    # let's just start with something simple...
    mvars = self.vars[len(CHANNELS) + 4:]
    #mvars = self.vars[4:]
    x = tf.concat([enc1, enc2], axis=-1)
    w,b = mvars[:2]
    distance= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[2:4]
    distance = tf.einsum('ba,ao->bo',distance,w) + b
    return distance

  @tf.function(input_signature=(IMSPEC, IMSPEC))
  def distance_states(self, s1, s2):
    enc1 = self.encode(s1)
    enc2 = self.encode(s2)
    return self.distance(enc1, enc2)


  @tf.function(input_signature=(IMSPEC, IMSPEC, IMSPEC, INTSPEC, FLOATSPEC, BOOLSPEC))
  def loss(self, states_a, states_b, states_k, action, probs, dones_ab):

    enca, encb, enck = [self.encode(x) for x in [states_a, states_b, states_k]]
    
    Dak = self.distance(enca, enck)
    Dak_a = tf.gather(Dak, action, batch_dims=1)
    Dab = self.distance(enca, encb)
    self.Dab = Dab
    Dab_a = tf.gather(Dab, action, batch_dims=1)
    self.Dab_a = Dab_a
    Dbk_target = self.distance(encb, enck)

    # check if k == b. If so, target Dbk needs to be 0
    target = tf.reduce_min(Dbk_target, axis=-1)
    mask = tf.squeeze(tf.reduce_max(tf.cast(tf.not_equal(encb, enck), FLOAT_TYPE), axis=1))
    
    if not USE_LOG:
      target = tf.stop_gradient(1 + DISCOUNT * target * mask)
      Dab_target = 1
      max_target = 1 / (1 - DISCOUNT)
    else:
      target = tf.stop_gradient(tf.math.log(1 + mask * tf.math.exp(target)))
      Dab_target = 0
      max_target = tf.math.log(MAX_DIST)

    target = target - tf.cast(dones_ab, FLOAT_TYPE) * (target - max_target)

    # apply importance sampling to the target
    weights = tf.pow(probs, -BETA)
    weights /= tf.reduce_max(weights)

    TD_error = tf.abs(Dak_a - target) 
    TD_error = tf.pow(TD_error, ALPHA)


    loss_TD = tf.reduce_mean(tf.pow(weights * (Dak_a - target), 2))
    # TODO having this update in here is going to mess things up when 
    # transitions are stochastic 
    loss_ab = tf.reduce_mean(tf.pow(weights * (Dab_a - Dab_target), 2))

    regloss = 0
    for x in self.vars:
      regloss += tf.reduce_mean(tf.pow(x, 2))

    if not USE_LOG:
      av_distance = tf.reduce_mean(Dak)
      max_distance = tf.reduce_max(Dak)
    else:
      av_distance  = tf.reduce_mean(tf.math.exp(Dak))
      max_distance = tf.reduce_max(tf.math.exp(Dak))

    
    loss = AB_WEIGHT * loss_ab + loss_TD + REGULARIZATION_WEIGHT * regloss
    grads = tf.gradients(loss, self.vars)
    self.opt.apply_gradients(zip(grads, self.vars))

    return (loss_TD, loss_ab, regloss), TD_error, av_distance

    
if __name__ == '__main__':

  agent = Agent();

  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)

  save = {}
  with open(picklepath, "wb") as fp:
    pickle.dump(save, fp)
    

    
    




    
