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
from preference_learning import Model as RewardModel

class EnsembleModel():
    def __init__(self, env, num_models, model_dir):
        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                self.models = []
                for i in range(num_models):
                    with tf.variable_scope('model_%d'%i):
                        model = RewardModel(env.observation_space.shape[0])
                        model.saver.restore(self.sess,model_dir+'/model_%d.ckpt'%(i))
                    self.models.append(model)

    def reward(self,ob):
        with self.graph.as_default():
            with self.sess.as_default():
                r_hat = 0.
                for model in self.models:
                    r_hat += model.get_reward(ob[None])[0]
                return r_hat

def gen_traj(env,agent,reward_model,render=False):
    ob = env.reset()
    true_r = my_r = 0.
    done = False

    while not done:
        a = agent.act(ob,None,None)
        ob, r, done, _ = env.step(a)

        true_r += r
        my_r += reward_model.reward(ob)

        if render:
            env.render()

    print('last_x_loc', env.env.sim.data.qpos[0],'true r',true_r,'my r',my_r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--agents_path', default='', help='path of learning agents')
    parser.add_argument('--reward_model_path', default='', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    reward_model = EnsembleModel(env,args.num_models,args.reward_model_path)

    models = sorted(Path(args.agents_path).glob('?????'))[::-1]
    assert len(models) > 0, 'no agent is given'

    for i,path in enumerate(models):
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=False)

        print(path)
        gen_traj(env,agent,reward_model,args.render)


