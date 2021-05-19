


# Goal-directed reinforcement learning

The ultimate goal of this project is reinforcement learning
with goals (where a goalsetter sets a goal, and then the agent tries to achieve
that goal). The motivation for this is that goal-setting seems to me to be a better approximation
of how humans think than the main alternatives, which mostly fall into two broad categories - "reflex" 
learning, where a function approximator (i.e. neural network) learns how to map states to actions directly, 
and "tree search" methods, where a learned dynamics model is used to search the decision tree for
the best actions.  For example, Q-learning [3] and Agent57 [5] are examples of "reflex" learning, while
MuZero [4] is an example of "tree search" learning.

While humans certainly demonstrate both of these "reflex" and "tree search" capabilities, in my opinion, most higher-level
thought takes the form of goal-setting, goal-evaluating, and goal-pursuing. There are many other
examples of goal-oriented reinforcement learning ([1], [2]), but my approach here is
unique enough that I thought it worth exploring.

## Goal-pursuing

Goal-oriented AI requires two capabilities: a way to set goals (the goal setter), and a way to achieve
goals (the goal pursuer). In this project so far I have only focused on the goal pursuer. My initial
idea was that if asymmetric distance function could be learned between states, the goal pursuer could
be trained with intrinsic rewards for moving closer to the goal state. However, I soon realized that
a simpler approach could allow simultaneous learning of the distance function and the 
'actor' that moves from one state to another: if we represent the distance function
D(s,g,a) => R
as a funciton mapping current state (s), goal state (g), and action (a) to a distance, where D(s,g,a) = (distance from s to g after taking action a),
then this function can be learned in a similar manner to standard Q-learning: for each transition (s1,s2,a) with goal state (g),
we can use the update
D(s1,g,a) <= 1 + gamma * min b (D(s2, g, b))
This is a contraction in the sup-norm, just like Q-learning. See section TODO for a proof of the contraction
for the final version of this function (I modify it in the "dead end toy environment" section)
Then, the goal pursuer can simply pick the action a that minimizes the predicted distance to the goal!
In practice I suspect things will be a little more complicated than this - if the action with minimum distance
is picked every time, then it's possible that "mistakes" in the learned distance function will lead to the goal
pursuer getting stuck in loops. This problem could potentially be solved by having the goal pursuer randomly sample
from actions according D(s1,sk,a) (so actions that lead to large distances would be less likely than those that lead to
small distances).

## Basic Grid Environment
To test the basic feasibility of this idea, I started with a simple 20x20 grid environment, where the state is
a pair of integer coordinates in 20x20 space, and the 5 actions 
correspond to moving in the cardinal directions, and waiting in place. Additionally, if the agent
tries to move off the edge of the grid, it remains in place. A great advantage of such a toy environment
is that the accurracy of the learned distance function, and the accurracy of the actions it implies for a goal pursuer, 
can both be easily measured. The below figures show the training curves of action accurracy and distance accurracy 
for the 20x20 environment.

TODO include figures

An obvious next step is to increase the size of the toy environment. In my experiments I have not yet been able
to reach 100% action accurracy in a toy environment above the size of about 40x40 (see discussion section, TODO)


## Uncommon transition toy environment (UTTE) and Prioritized Experience Replay:
Clearly, the basic toy environment described above lacks many characteristics of more interesting environments.
One such characteristic is seldom-seen transitions. The UTTE is the same as the basic grid environment, with 
the exception that transitions off the right-hand side of the grid bring the agent back around to the left hand side.
Since the agent so far learns the distance function entirely through random actions, these transitions happen relatively
infrequently (the agent's random walk needs to bring it all the way across the grid). Without any modifications to the
above approach, the distance function is not learned on this environment. So, I added prioritized experience replay,
a common and successful technique to deal with this 'problem' of rarely-seen transitions. Below are the graphs
for the learned action accurracy with and without prioritized experience replay.

TODO include figures

## Dead end toy environment (DETE):
My ultimate goal for this project was to train the agent in the Montezuma's Revenge Atari environment.
One important characteristic of this environment is that it has many "dead ends" - for example, when the
player jumps off a ledge, they are stuck in "dead end" until they hit the ground, die, and the game restarts.
To replicate this scenario, the DETE is a simple 1d line of 10 states, where each action transitions the
agent one state left, and when the agent reaches the last state, the environment is terminated. Hence,
the distance function must learn maximal distances for any (state, goal) pairs where the goal
happens before the state in this deterministic progression (the maximal distance is limited by the gamma parameter).

![proof of contraction of log-distance update](/images/contraction_proof.pdf)


## Network and implementation details
The distance function is learned with a neural network approximator. On toy environments, states are represented by
tuples of integers (x,y). The neural network learns an "encoding" of the state with two
fully connected layers, each with output size 128 and activation leaky relu. The 
distance between two states is taken by concatenating their two encodings, and passing
this through two fully connected layers, the first with output size 128, and
the second with output size |A| (the number of actions).  

The loss for the log distance function is computed as follows: for each transition
(s1, s2, a), with goal state (sk), the Dab loss is (D(s1, s2, a))^2, and
the Dak loss is (D(s1, sk, a) - log (1 + min_b e^D(s2, sk, b)))^2. If 
s1 -> s2 is a terminal transition, then the target D(s2, sk, b) is set to be
the maximum possible (I used 10). For prioritized replay, I used an alpha
value of 1.0 and a beta of 0.0 (since I'm working in deterministic environments
only so far, importance sampling is not required for prioritized replay). The 
Dab and Dak losses are combined to get Loss = C * Dab + Dak. I left C = 1.0
through all my experiments. I used an experience replay of size 2^18,
the Adam optimizer with learning rate 0.001, and batch size of 128.  After filling
up the experience replay buffer, I trained with alternating steps of acting 
for 512 steps, and then training for 600 batches.

## Discussion
The major hurdle I'm facing at the moment is getting the model to learn larger
environments (it's stuck on the 80x80 basic toy environment). I've tried training
it on the Montezuma's Revenge deterministic Atari environment a few times, 
upgrading the encoding step to use CNNs instead of fully connected layers, and it
is able to learn some dead-end states, but distances are mostly meaningless otherwise.
Most successfull deep reinforcement learning approaches use n-step rollouts for
the targets instead
of single-step TD errors, which is a downside of my approach (I don't see how 
to easily extend it to use n-step rollouts for distances).




References

[1] Universal Value Function Approximators, Schaul et al, 2015.
http://proceedings.mlr.press/v37/schaul15.pdf

[2] Hindsight Experience Replay, Andrychowicz et al, 2017.
https://arxiv.org/pdf/1707.01495.pdf

[3] Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

[4] Mastering Atari, Go, Chess, and Shogi by Planning with a Learned Model, Schrittwieser et al, 2019.
https://arxiv.org/pdf/1911.08265.pdf

[5] Agent57: Outperforming the Atari Human Benchmark, Badia et al, 2020.
https://arxiv.org/pdf/2003.13350v1.pdf


