import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import Monitor
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

from siamese_ranker import PPO2Agent

def gen_traj(env,agent,render=False,max_len=99999):
    ob = env.reset()

    for _ in range(max_len):
        a = agent.act(ob,None,None)
        ob, r, done, _ = env.step(a)

        if render:
            env.render()
        if done:
            break

    try:
        return env.env.sim.data.qpos[0] #Final location of the object
    except AttributeError:
        return env.env.env.sim.data.qpos[0] #Final location of the object

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--agents_path', default='', help='path of learning agents')
    parser.add_argument('--num_trajs', default=10, type=int, help='path of learning agents')
    parser.add_argument('--max_len', default=1000, type=int)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    models = sorted(Path(args.agents_path).glob('?????'))[::-1]

    best_perfs = -9999999
    for i,path in enumerate(models):
        model_name = path.name
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)

        perfs = []
        for j in range(args.num_trajs):
            if j == 0 and args.video_record:
                wrapped = Monitor(env, './video/%s'%model_name,force=True)
            else:
                wrapped = env

            perfs.append(gen_traj(wrapped,agent,args.render,args.max_len))

        print(path, ',', np.mean(perfs), ',', np.std(perfs))

        if np.mean(perfs) > best_perfs:
            best_perfs = np.mean(perfs)

    print('best performance', best_perfs)
