import argparse
import tensorflow as tf
import numpy as np
import gym
from tf_commons.ops import *

from siamese_ranker import PPO2Agent
from behavior_cloning import Dataset, Policy
from performance_checker import gen_traj

"""
idea here is that since the inferred reward function is differentiable, why don't we directly optimize a policy for that? It can be done by SGD, or some optimization algorithm.
It would be nice if the reward function is convex, but currently not.

Currently, it doesn't seem like working, but we might have to dig more
"""

class Model(object):
    def __init__(self,ob_dim,ac_dim,embedding_dims=256):
        in_dims = ob_dim+ac_dim

        self.state = tf.placeholder(tf.float32,[None,ob_dim])
        self.action = tf.placeholder(tf.float32,[None,ac_dim])

        with tf.variable_scope('weights') as param_scope:
            self.fc1 = Linear('fc1',in_dims,embedding_dims)
            self.fc2 = Linear('fc2',embedding_dims,embedding_dims)
            self.fc3 = Linear('fc3',embedding_dims,1)

        self.param_scope = param_scope

        # build graph
        def _reward(x):
            _ = tf.nn.relu(self.fc1(x))
            _ = tf.nn.relu(self.fc2(_))
            r = tf.squeeze(self.fc3(_),axis=1)
            return r

        self.r = _reward(tf.concat([self.state,self.action],axis=1))
        self.action_grad = tf.gradients([self.r],[self.action])[0]

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def enhance_demo(self,b_s,b_a,lr,steps):
        sess = tf.get_default_session()

        b_a = np.copy(b_a)
        for s in range(steps):
            grad = sess.run(self.action_grad,feed_dict={self.state:b_s,self.action:b_a})
            b_a += lr * grad

        return b_a

def train(args):
    env = gym.make(args.env_id)
    with tf.variable_scope('model_0'):
        model = Model(env.observation_space.shape[0],env.action_space.shape[0])

    # Training configuration
    sess = tf.InteractiveSession()
    model.saver.restore(sess,args.reward_model_path)

    demo_agent = PPO2Agent(env,args.env_type,str(args.learner_path))

    dataset = Dataset(env)
    dataset.prebuilt([demo_agent],args.min_length)

    lr = 0.001
    steps = 1000

    learner = Policy(env.observation_space.shape[0],env.action_space.shape[0])

    obs,acs,_ = dataset.trajs
    for it in range(100):
        # train with current demo
        learner.sess.run(learner.init_op)
        learner.train((obs,acs,None),iter=20000)

        # evalutate
        last_pos = [gen_traj(env,learner) for _ in range(10)]
        print('[%d] %f (%f)'%(it,np.mean(last_pos),np.std(last_pos)))

        # enhance demo
        acs = model.enhance_demo(obs,acs,lr,steps)

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='Ant-v2', help='Select the environment to run')
    parser.add_argument('--env_type', default='mujoco', help='mujoco or atari')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--learner_path', default='./learner/demo_models/ant/checkpoints/00305', help='path to learning agent')
    parser.add_argument('--reward_model_path', default='./log/ant_gt/model_0.ckpt', help='path to a reward model')
    parser.add_argument('--max_len', default=1000, type=int)
    args = parser.parse_args()

    train(args)
