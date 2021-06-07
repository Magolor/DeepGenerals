## Open AI Gym

#### Environment

* `(observation, reward, done, info) = gym.env.step(action)`:  respons for given action
  * observation: depands on game type
  * reward: a float
  * done: a bool
  * info: any auxillary information
* `gym.env.reset()`:  restart the game
* `gym.env.render()`: visualization

#### Observations

Environment, agent specific information for the state of environment. Four tuple:
$$
(obser, reward, done, info)
$$


#### Space

* Box(): cartesian product of intervals
* Discrete(): a fixed number of actions
* MultiDiscrete(): a fixed number of action for multiple dimension, i.e. chess.
* MultiBinary(): for fixed two choices in each dimension
*  Dict & tuple: compose simple space

###### Action Space

* sample()
* contains()

###### Observation Space

* sample()
* contains()