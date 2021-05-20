import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from train import *

import agent


dir_path = os.path.dirname(os.path.realpath(__file__))

model_savepath = os.path.join(dir_path, 'saved_for_graphs/loopenv_nolog_nopriority')
model_savepath = os.path.join(dir_path, 'saved_for_graphs/loopenv_nolog_priority')
model_savepath = os.path.join(dir_path, 'saved_for_graphs/20x20_nolog_1024bs')
model_savepath = os.path.join(dir_path, 'actor_save')

model_savepath = os.path.join(dir_path, 'saved_for_graphs/deadend10_1024_nolog')
#model_savepath = os.path.join(dir_path, 'saved_for_graphs/deadend_withlog_128_moddist1')
model_savepath = os.path.join(dir_path, 'saved_for_graphs/deadend_log_1024')
picklepath = os.path.join(model_savepath, 'actor.pickle')


with open(picklepath, "rb") as f: 
  save = pickle.load(f)
  #accs = save['accs']
  dist_diffs = save['dist_diffs']
  #save_stats_every = save['save_stats_every']



STEPS_TO_GRAPH = 20000
steps = np.arange(0, STEPS_TO_GRAPH, SAVE_STATS_EVERY)

steps = steps[: STEPS_TO_GRAPH//SAVE_STATS_EVERY]
dist_diffs = dist_diffs[: STEPS_TO_GRAPH//SAVE_STATS_EVERY]
#accs = accs[: STEPS_TO_GRAPH//SAVE_STATS_EVERY]

fig, ax = plt.subplots()

ax.set_title('Log-distance on DeadEnd env')

#twin1 = ax.twinx()

p1, = ax.plot(steps, dist_diffs, 'b-', label='Mean |distance difference| on reachable state/goal pairs')
#p2, = ax.plot(log, 'r-', label='Log distance')
ax.grid(True)

ax.set_xlabel('Parameter updates')
ax.set_ylabel('Mean |distance difference| on reachable state/goal pairs')
#twin1.set_ylabel('Distance errors')

#tkw = dict(size=4, width=1.5)
#ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
#twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
#ax.tick_params(axis='x', **tkw)

#ax.yaxis.label.set_color(p1.get_color())
#twin1.yaxis.label.set_color(p2.get_color())

start, end = ax.get_ylim()
#ax.set_ylim(0,0.1)
#ax.yaxis.set_ticks(np.arange(0, 0.1, 0.001))
#ax.set_yscale('log')
plt.show()
