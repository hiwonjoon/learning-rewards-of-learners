import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

from siamese_ranker import PPO2Agent

def gen_traj(env,agent,render=False):
    ob = env.reset()
    done = False

    while not done:
        a = agent.act(ob,None,None)
        ob, r, done, _ = env.step(a)

        if render:
            env.render()

    return env.env.sim.data.qpos[0] #Final location of the object

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--agents_path', default='', help='path of learning agents')
    parser.add_argument('--num_trajs', default=10, help='path of learning agents')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    models = sorted(Path(args.agents_path).glob('?????'))[::-1]

    best_perfs = -9999999
    for i,path in enumerate(models):
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=False)

        perfs = []
        for _ in range(args.num_trajs):
            perfs.append(gen_traj(env,agent,args.render))

        print(path, ',', np.mean(perfs), ',', np.std(perfs))

        if np.mean(perfs) > best_perfs:
            best_perfs = np.mean(perfs)

    print('best performance', best_perfs)
