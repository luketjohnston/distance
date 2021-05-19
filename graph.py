import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from train import *

dir_path = os.path.dirname(os.path.realpath(__file__))
loss_savepath = os.path.join(dir_path, 'actor.pickle')

import agent


with open(agent.picklepath, "rb") as f: 
  save = pickle.load(f)
  accs = save['accs']
  dist_diffs = save['dist_diffs']
  #save_stats_every = save['save_stats_every']



steps = np.arange(0, SAVE_STATS_EVERY*len(accs), 50)

fig, ax = plt.subplots()

twin1 = ax.twinx()

p1, = ax.plot(steps, accs, 'b-', label='Action accurracy')
p2, = twin1.plot(steps, dist_diffs, 'r-', label='Distance errors')
ax.grid(True)

ax.set_xlabel('Parameter updates')
ax.set_ylabel('Action accurracy')
twin1.set_ylabel('Distance errors')

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())

start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
