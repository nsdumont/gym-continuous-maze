import gymnasium as gym
import gym_continuous_maze
import numpy as np

env = gym.make('ContinuousMaze-5x5-v0')
obs,_ = env.reset()
obs,r,t1,t2,_ = env.step(np.array([1,0]))
obs,r,t1,t2,_ = env.step(np.array([1,0]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([1,0]))
obs,r,t1,t2,_ = env.step(np.array([0,-1]))
obs,r,t1,t2,_ = env.step(np.array([0,-1]))
obs,r,t1,t2,_ = env.step(np.array([0,-1]))
obs,r,t1,t2,_ = env.step(np.array([1,0]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
obs,r,t1,t2,_ = env.step(np.array([0,1]))
