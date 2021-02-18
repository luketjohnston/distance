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
from toyenv import ToyEnv


# For some reason this is necessary to prevent error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ENVIRONMENT = 'MontezumaRevenge-v0'
ENVIRONMENT = 'PongDeterministic-v4'
#ENVIRONMENT = 'CartPole-v1'
env = gym.make(ENVIRONMENT)
env = ToyEnv(84)



ACTIONS = env.action_space.n

WIDTH = 84
HEIGHT = 110
DEPTH = 4

WIDTH = 84
HEIGHT = 84
DEPTH = 1


ENT_EPSILON = 1e-7

HIDDEN_NEURONS=128

REGULARIZATION_WEIGHT = 0

FILTER_SIZES = [8, 4, 3]
#FILTER_SIZES = []
CHANNELS =     [32,64,64]
#CHANNELS = []
STRIDES =     [4,2,1]
#STRIDES = []

ENCODING_SIZE = 2


IMSPEC = tf.TensorSpec([None, WIDTH, HEIGHT, DEPTH])
if ENVIRONMENT == 'CartPole-v1':
  IMSPEC = tf.TensorSpec([None, 4])
# FOR distance proof - of - concept
IMSPEC = tf.TensorSpec([None, 2,1])

INTSPEC = tf.TensorSpec([None], dtype=tf.int64)
FLOATSPEC = tf.TensorSpec([None],)
DISTSPEC = tf.TensorSpec([None, ACTIONS],)
BOOLSPEC = tf.TensorSpec([None], dtype=tf.bool)
LOGITSPEC = tf.TensorSpec([None, ACTIONS])
ENCSPEC = tf.TensorSpec([None, ENCODING_SIZE])


DISCOUNT = 0.999
#DISCOUNT = 0.99
ENTROPY_WEIGHT = 0.001
EPSILON = 0.1

#ADD_ENTROPY = True

dir_path = os.path.dirname(os.path.realpath(__file__))
model_savepath = os.path.join(dir_path, 'actor_save2')
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


    self.vars.append(tf.Variable(tf.initializers.GlorotNormal()(shape=(2,HIDDEN_NEURONS)), name='Encoder_w'))
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
    #for i in range(len(CHANNELS)):
    #  filt = mvars[i]
    #  stride = STRIDES[i]
    #  x = tf.nn.conv2d(x,filt,stride,'VALID',name=None)
    #  x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Flatten()(x)
    #vi = len(CHANNELS)
    #vi = 0
    #w, b = mvars[vi:vi+2]
    #encoding= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    #w,b = mvars[vi+2:vi+4]
    #encoding = tf.nn.leaky_relu(tf.squeeze(tf.einsum('ba,ao->bo',encoding,w))  + b)
    return x

  ''' Takes two state encodings as input, and returns the distance between them for each action taken a'''
  @tf.function(input_signature=(ENCSPEC,ENCSPEC))
  def distance(self, enc1, enc2):
    # let's just start with something simple...
    #mvars = self.vars[len(CHANNELS) + 4:]
    mvars = self.vars[4:]
    x = tf.concat([enc1, enc2], axis=-1)
    w,b = mvars[:2]
    distance= tf.nn.leaky_relu(tf.einsum('ba,ah->bh', x,w) + b)
    w,b = mvars[2:4]
    distance = tf.squeeze(tf.einsum('ba,ao->bo',distance,w))  + b
    # TODO should this be RELU instead?
    #tf.print('here')
    #tf.print(enc1[0,:])
    #tf.print(enc2[0,:])
    #r = tf.stack([tf.reduce_sum(tf.abs(enc1 - enc2), -1) for _ in range(ACTIONS)], -1)
    #tf.print(r)
    return distance

  @tf.function(input_signature=(IMSPEC, IMSPEC))
  def distance_states(self, s1, s2):
    enc1 = self.encode(s1)
    enc2 = self.encode(s2)
    return self.distance(enc1, enc2)


  @tf.function(input_signature=(IMSPEC, IMSPEC, IMSPEC, INTSPEC, DISTSPEC))
  def loss(self, states_a, states_b, states_k, action, Dbk_target):
    enca, encb, enck = [self.encode(x) for x in [states_a, states_b, states_k]]
    
    Dak = self.distance(enca, enck)
    Dak_a = tf.gather(Dak, action, batch_dims=1)
    #Dbk = self.distance(encb, enck)
    Dab = self.distance(enca, encb)
    Dab_a = tf.gather(Dab, action, batch_dims=1)

    #tf.print(Dak)
    #tf.print(Dab)
    #tf.print(states_k)

    # check if k == b. If so, target Dbk needs to be 0
    # TODO need to chance axis back to (1,2,3) for images
    target = Dbk_target * tf.expand_dims(tf.reduce_max(tf.cast(tf.not_equal(states_b, states_k), tf.float32), axis=(1)), -1)
    #tf.print(tf.equal(target, 0.0))
    target = tf.stop_gradient(1 + DISCOUNT * tf.reduce_min(target, axis=-1))
    loss = tf.reduce_mean(tf.pow(Dak_a - target, 2))
    #loss = tf.reduce_mean(tf.abs(Dak_a - target))

    loss += tf.reduce_mean(tf.pow(Dab_a - 1, 2))
    #loss += tf.reduce_mean(tf.abs(Dab_a - 1))

    regloss = 0
    for x in self.vars:
      #regloss += REGULARIZATION_WEIGHT * tf.reduce_sum(tf.pow(x, 2))
      regloss += REGULARIZATION_WEIGHT * tf.reduce_sum(tf.abs(x))


    return (loss, regloss)
    
    
if __name__ == '__main__':

  #print('running loss')
  #encoder.loss_from_beginning(tf.zeros((16,84,110,1)))

  

  agent = Agent();

  print('Saving model...')
  tf.saved_model.save(agent, model_savepath)


  losses = []
  episode_rewards = []
  with open(loss_savepath, "wb") as fp:
    pickle.dump(losses, fp)
  with open(rewards_savepath, "wb") as fp:
    pickle.dump(episode_rewards, fp)
    

    
    




    
