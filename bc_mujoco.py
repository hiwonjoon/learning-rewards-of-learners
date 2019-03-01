import os
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

from tf_commons.ops import *
from siamese_ranker import PPO2Agent

class InverseDynamics(object):
    def __init__(self,ob_dim,ac_dim,num_layers=4,embed_size=256):
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            self.s_ns = tf.placeholder(tf.float32,[None,ob_dim*2])
            self.a = tf.placeholder(tf.float32,[None,ac_dim])
            self.l2_reg = tf.placeholder(tf.float32,[])

            with tf.variable_scope('weights') as param_scope:
                self.param_scope = param_scope

                fcs = []
                last_dims = ob_dim*2
                for l in range(num_layers):
                    fcs.append(Linear('fc%d'%(l+1),last_dims,embed_size)) #(l+1) is gross, but for backward compatibility
                    last_dims = embed_size
                fcs.append(Linear('fc%d'%(num_layers+1),last_dims,ac_dim))

            # build graph
            def _build(x):
                for fc in fcs[:-1]:
                    x = tf.nn.relu(fc(x))
                pred_a = fcs[-1](x)
                return pred_a

            self.pred_a = _build(self.s_ns)
            self.loss = tf.reduce_mean(tf.reduce_sum((self.pred_a - self.a)**2,axis=1))

            weight_decay = 0.
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.w**2)

            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

            self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self,train=False):
        with self.graph.as_default():
            if train:
                return tf.trainable_variables(self.param_scope.name)
            else:
                return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size,iter,l2_reg,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs)-1)
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append((np.concatenate([obs[i],obs[i+1]]),acs[i]))

            b_s_ns,b_a = zip(*batch)
            b_s_ns,b_a = np.array(b_s_ns),np.array(b_a)

            return b_s_ns,b_a

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_s_ns,b_a = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,_ = sess.run([self.loss,self.l2_loss,self.update_op],feed_dict={
                    self.s_ns:b_s_ns,
                    self.a:b_a,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_s_ns,b_a= _batch(valid_idxes)
                    valid_loss = sess.run(self.loss,feed_dict={
                        self.s_ns:b_s_ns,
                        self.a:b_a,
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f'%(loss,l2_loss,valid_loss)))

    def infer_action(self, b_obs, batch_size=64):
        sess = self.sess

        with self.graph.as_default():
            ac = []
            for s in range(0,len(b_obs)-1,batch_size):
                pred_a = sess.run(self.pred_a,feed_dict={
                    self.s_ns:np.concatenate([b_obs[s:min(s+batch_size,len(b_obs)-1)],b_obs[s+1:min(s+1+batch_size,len(b_obs))]],axis=1)})
                ac.append(pred_a)

        return np.concatenate(ac,axis=0)

    def save(self,path):
        with self.graph.as_default():
            self.saver.save(self.sess,path,write_meta_graph=False)

    def load(self,path):
        with self.graph.as_default():
            self.saver.restore(self.sess,path)

class Policy(object):
    def __init__(self,ob_dim,ac_dim,num_layers,embed_size):
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.inp = tf.placeholder(tf.float32,[None,ob_dim])
            self.l = tf.placeholder(tf.float32,[None,ac_dim])
            self.l2_reg = tf.placeholder(tf.float32,[])

            with tf.variable_scope('weights') as param_scope:
                self.param_scope = param_scope

                fcs = []
                last_dims = ob_dim
                for l in range(num_layers):
                    fcs.append(Linear('fc%d'%(l+1),last_dims,embed_size)) #(l+1) is gross, but for backward compatibility
                    last_dims = embed_size
                fcs.append(Linear('fc%d'%(num_layers+1),last_dims,ac_dim))

            # build graph
            def _build(x):
                for fc in fcs[:-1]:
                    x = tf.nn.relu(fc(x))
                pred_a = fcs[-1](x)
                return pred_a

            self.ac = _build(self.inp)

            loss = tf.reduce_sum((self.ac-self.l)**2,axis=1)
            self.loss = tf.reduce_mean(loss,axis=0)

            weight_decay = 0.
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.w**2)

            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

            self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self,train=False):
        with self.graph.as_default():
            if train:
                return tf.trainable_variables(self.param_scope.name)
            else:
                return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size,iter,l2_reg,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs)-1)
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            batch = []
            for i in idxes:
                batch.append((obs[i],acs[i]))
            b_s,b_a = [np.array(e) for e in zip(*batch)]

            return b_s,b_a

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_s,b_a = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,_ = sess.run([self.loss,self.l2_loss,self.update_op],feed_dict={
                    self.inp:b_s,
                    self.l:b_a,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_s,b_a= _batch(valid_idxes)
                    valid_loss = sess.run(self.loss,feed_dict={
                        self.inp:b_s,
                        self.l:b_a,
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f'%(loss,l2_loss,valid_loss)))

    def act(self, observation, reward, done):
        sess = self.sess

        with self.graph.as_default():
            ac = sess.run(self.ac,feed_dict={self.inp:observation[None]})[0]

        return ac

    def save(self,path):
        with self.graph.as_default():
            self.saver.save(self.sess,path,write_meta_graph=False)

    def load(self,path):
        with self.graph.as_default():
            self.saver.restore(self.sess,path)

class Dataset(object):
    def __init__(self,env):
        self.env = env

    def gen_traj(self,agent,min_length):
        obs, actions, rewards = [self.env.reset()], [], []

        # For debug purpose
        last_episode_idx = 0
        acc_rewards = []

        while True:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                if len(obs) < min_length:
                    obs.pop()
                    obs.append(self.env.reset())

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                else:
                    obs.pop()

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                    break

        return np.stack(obs,axis=0), np.stack(actions,axis=0), np.array(rewards), np.mean(acc_rewards)

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            (*traj), avg_acc_reward = self.gen_traj(agent,min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f'%(agent.model_path,avg_acc_reward))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape,self.trajs[2].shape)

    def infer_action(self,inv_model):
        obs, _, rewards = self.trajs
        acs = inv_model.infer_action(obs)
        self.trajs = (obs,acs,rewards)

def setup_logdir(args):
    logdir = Path(args.log_path)
    if logdir.exists() :
        c = input('log is already exist. continue [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
    logdir.mkdir(parents=True)
    with open(str(logdir/'args.txt'),'w') as f:
        f.write( str(args) )
    return str(logdir)

def build_dataset(args,env):
    dataset = Dataset(env)

    demo_agents = []
    demo_chkpt = eval(args.demo_chkpt)
    if type(demo_chkpt) == int:
        demo_chkpt = list(range(demo_chkpt+1))
    else:
        demo_chkpt = list(demo_chkpt)

    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) in demo_chkpt])
    for path in models:
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
        demo_agents.append(agent)

    dataset.prebuilt(demo_agents,args.min_length)
    return dataset

def train_inv(args):
    logdir = setup_logdir(args)

    env = gym.make(args.env_id)

    dataset = build_dataset(args,env)

    inv_dy = InverseDynamics(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)

    D = dataset.trajs

    inv_dy.train(D,args.batch_size,args.num_iter,l2_reg=args.l2_reg,debug=True)
    inv_dy.save(os.path.join(logdir,'model.ckpt'))

def bc(args):
    logdir = setup_logdir(args)

    env = gym.make(args.env_id)

    dataset = build_dataset(args,env)

    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)

    D = dataset.trajs
    policy.train(D,args.batch_size,args.num_iter,l2_reg=args.l2_reg,debug=True)
    policy.save(os.path.join(logdir,'model.ckpt'))

def bco(args):
    logdir = setup_logdir(args)

    env = gym.make(args.env_id)

    dataset = build_dataset(args,env)

    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)
    inv_dy = InverseDynamics(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)
    inv_dy.load(args.inv_model)

    dataset.infer_action(inv_dy)

    D = dataset.trajs
    policy.train(D,args.batch_size,args.num_iter,l2_reg=args.l2_reg,debug=True)
    policy.save(os.path.join(logdir,'model.ckpt'))

def eval_(args):
    env = gym.make(args.env_id)

    logdir = str(Path(args.log_path))
    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    policy.load(os.path.join(logdir,'model.ckpt'))

    from performance_checker import gen_traj
    from gym.wrappers import Monitor
    perfs = []
    for j in tqdm(range(args.num_tries)):
        if j == 0 and args.video_record:
            wrapped = Monitor(env, './video/',force=True)
        else:
            wrapped = env

        perfs.append(gen_traj(wrapped,policy,args.render,99999))
    print(logdir, ',', np.mean(perfs), ',', np.std(perfs))

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    # Train Setting
    parser.add_argument('--mode', required=True, choices=['train_inv','bc','bco','eval'])
    parser.add_argument('--log_path', required=True, help='path to log base (mode & env_id will be concatenated at the end)')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--env_type', default='mujoco', help='mujoco or atari', choices=['mujoco','atari'])
    # Dataset
    parser.add_argument('--learners_path', required=True, help='path of learning agents')
    parser.add_argument('--demo_chkpt', default='240', help='decide upto what learner stage you want to give')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    # Network
    parser.add_argument('--num_layers', default=4,type=int)
    parser.add_argument('--embed_size', default=256,type=int)
    # Training
    parser.add_argument('--l2_reg', default=0.001, type=float, help='l2 regularization size')
    parser.add_argument('--inv_model', default='') # For BCO only
    parser.add_argument('--num_iter',default=50000,type=int)
    parser.add_argument('--batch_size',default=128,type=int)
    # For Eval
    parser.add_argument('--num_tries', default=10, type=int, help='path of learning agents')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()

    if args.mode == 'train_inv':
        train_inv(args)
    elif args.mode == 'bc':
        bc(args)
    elif args.mode == 'bco':
        bco(args)
    elif args.mode == 'eval':
        eval_(args)
    else:
        assert False
