import os
import argparse
from functools import partial
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import pynvml as N

from tf_commons.ops import *
from siamese_ranker import PPO2Agent

# TODO: merge train / train_with_dataset

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()[None]


################################

class Net(object):
    def input_preprocess(self,obs,acs):
        # Be careful when implementing this.
        # This function have to process raw input of obs and acs in the same manner as Dataset class does.
        raise NotImplementedError

    def build_input_placeholder(self,name):
        raise NotImplementedError

    def build_reward(self,x):
        raise NotImplementedError

    def build_weight_decay(self):
        raise NotImplementedError

class MujocoNet(Net):
    def __init__(self,include_action,ob_dim,ac_dim,num_layers=2,embedding_dims=256):
        in_dims = ob_dim+ac_dim if include_action else ob_dim

        with tf.variable_scope('weights') as param_scope:
            fcs = []
            last_dims = in_dims
            for l in range(num_layers):
                fcs.append(Linear('fc%d'%(l+1),last_dims,embedding_dims)) #(l+1) is gross, but for backward compatibility
                last_dims = embedding_dims
            fcs.append(Linear('fc%d'%(num_layers+1),last_dims,1))

        self.fcs = fcs
        self.in_dims = in_dims
        self.include_action = include_action
        self.param_scope = param_scope

    def input_preprocess(self,obs,acs):
        return \
            np.concatenate((obs,acs),axis=1) if self.include_action \
            else obs

    def build_input_placeholder(self,name):
        return tf.placeholder(tf.float32,[None,self.in_dims],name=name)

    def build_reward(self,x):
        for fc in self.fcs[:-1]:
            x = tf.nn.relu(fc(x))
        r = tf.squeeze(self.fcs[-1](x),axis=1)
        return x, r

    def build_weight_decay(self):
        weight_decay = 0.
        for fc in self.fcs:
            weight_decay += tf.reduce_sum(fc.w**2)
        return weight_decay

class AtariNet(Net):
    def __init__(self,ob_shape,embedding_dims=128):
        in_dims = list(ob_shape)

        with tf.variable_scope('weights') as param_scope:
            nets = []
            nets.append(Conv2d('conv1',4,32,k_h=8,k_w=8,d_h=4,d_w=4,data_format='NHWC',padding='VALID'))
            nets.append(Conv2d('conv2',32,64,k_h=4,k_w=4,d_h=2,d_w=2,data_format='NHWC',padding='VALID'))
            nets.append(Conv2d('conv3',64,64,k_h=3,k_w=3,d_h=2,d_w=2,data_format='NHWC',padding='VALID'))
            nets.append(Linear('fc1',64*4*4, embedding_dims))
            nets.append(Linear('fc2',embedding_dims, 1))

        self.nets = nets
        self.in_dims = in_dims
        self.param_scope = param_scope

    # build graph
    def input_preprocess(self,obs,acs):
        return obs

    def build_input_placeholder(self,name):
        return tf.placeholder(tf.float32,[None]+self.in_dims,name=name)

    def build_reward(self,x):
        for layer in self.nets[:-1]:
            x = tf.nn.relu(layer(x))
        r = tf.squeeze(self.nets[-1](x),axis=1)
        return x, r

    def build_weight_decay(self):
        weight_decay = 0.
        for layers in self.nets:
            weight_decay += tf.reduce_sum(layers.w**2)
        return weight_decay

class Model(object):
    def __init__(self,net:Net,batch_size=64):
        self.B = batch_size
        self.net = net

        self.x = net.build_input_placeholder('x') # tensor shape of [B*steps] + input_dims
        self.x_split = tf.placeholder(tf.int32,[self.B]) # B-lengthed vector indicating the size of each steps
        self.y = net.build_input_placeholder('y') # tensor shape of [B*steps] + input_dims
        self.y_split = tf.placeholder(tf.int32,[self.B]) # B-lengthed vector indicating the size of each steps
        self.l = tf.placeholder(tf.int32,[self.B]) # [0 when x is better 1 when y is better]

        self.l2_reg = tf.placeholder(tf.float32,[]) # [0 when x is better 1 when y is better]

        # Graph Ops for Inference
        self.fv, self.r = net.build_reward(self.x)

        # Graph ops for training
        _, rs_xs = net.build_reward(self.x)
        self.v_x = tf.stack([tf.reduce_sum(rs_x) for rs_x in tf.split(rs_xs,self.x_split,axis=0)],axis=0)

        _, rs_ys = net.build_reward(self.y)
        self.v_y = tf.stack([tf.reduce_sum(rs_y) for rs_y in tf.split(rs_ys,self.y_split,axis=0)],axis=0)

        logits = tf.stack([self.v_x,self.v_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        # Regualarizer Ops
        weight_decay = net.build_weight_decay()
        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.v_y,self.v_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.net.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.net.param_scope.name)

    def train(self,D,iter=10000,l2_reg=0.01,noise_level=0.1,debug=False,early_term=False):
        """
        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
                while
                    sigma^{1,2}: shape of [steps,in_dims]
                    mu : 0 or 1
            l2_reg
            noise_level: input label noise to prevent overfitting
            debug: print training statistics
            early_term:  training will be early-terminate when validation accuracy is larger than 95%
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D)*0.8)]
        valid_idxes = idxes[int(len(D)*0.8):]

        def _load(idxes,add_noise=True):
            batch = []

            for i in idxes:
                batch.append(D[i])

            b_x,b_y,b_l = zip(*batch)
            x_split = np.array([len(x) for x in b_x])
            y_split = np.array([len(y) for y in b_y])
            b_x,b_y,b_l = np.concatenate(b_x,axis=0),np.concatenate(b_y,axis=0),np.array(b_l)

            if add_noise:
                b_l = (b_l + np.random.binomial(1,noise_level,self.B)) % 2 #Flip it with probability 0.1

            return b_x.astype(np.float32),b_y.astype(np.float32),x_split,y_split,b_l


        def _batch(idx_list,add_noise):
            if len(idx_list) > self.B:
                idxes = np.random.choice(idx_list,self.B,replace=False)
            else:
                idxes = idx_list

            return _load(idxes,add_noise)

        def load(idxes):
            b_x,b_y,x_split,y_split,b_l =\
                tf.py_func(_load, [idxes], [tf.float32,tf.float32,tf.int64,tf.int64,tf.int64], stateful=False)
            b_x = tf.reshape(b_x,[-1,84,84,4])
            b_y = tf.reshape(b_y,[-1,84,84,4])
            x_split = tf.reshape(x_split,[self.B])
            y_split = tf.reshape(y_split,[self.B])
            b_l = tf.reshape(b_l,[self.B])

            return b_x,b_y,x_split,y_split,b_l

        ds = tf.data.Dataset.range(len(D))
        ds = ds.repeat(-1)
        ds = ds.shuffle(len(D))
        ds = ds.batch(64)
        ds = ds.map(load, num_parallel_calls=8)
        ds = ds.prefetch(10)

        batch_op = ds.make_one_shot_iterator().get_next()

        """
        def gen(idx_list,add_noise):
            while True:
                yield _batch(idx_list,add_noise)
        ds = tf.data.Dataset.from_generator(
            partial(gen,train_idxes,True),
            (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([None,84,84,4]), tf.TensorShape([None,84,84,4]),
            tf.TensorShape([self.B]),tf.TensorShape([self.B]),tf.TensorShape([self.B])))
        ds = ds.prefetch(buffer_size=10)
        batch_op = ds.make_one_shot_iterator().get_next()
        """

        for it in tqdm(range(iter),dynamic_ncols=True):
            #b_x,b_y,x_split,y_split,b_l = _batch(train_idxes,add_noise=True)
            b_x,b_y,x_split,y_split,b_l = sess.run(batch_op)

            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.x_split:x_split,
                self.y_split:y_split,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_x,b_y,x_split,y_split,b_l = _batch(valid_idxes,add_noise=False)
                    valid_acc = sess.run(self.acc,feed_dict={
                        self.x:b_x,
                        self.y:b_y,
                        self.x_split:x_split,
                        self.y_split:y_split,
                        self.l:b_l
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))

            if early_term and valid_acc >= 0.95:
                print('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc))
                print('early termination@%08d'%it)
                break

    def train_with_dataset(self,dataset,iter=10000,l2_reg=0.01,debug=False):
        sess = tf.get_default_session()

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_x,b_y,x_split,y_split,b_l = dataset.batch(batch_size=self.B,include_action=self.net.include_action)
            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.x_split:x_split,
                self.y_split:y_split,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f'%(loss,l2_loss,acc)))

    def get_reward(self,obs,acs,batch_size=1024):
        sess = tf.get_default_session()

        inp = self.net.input_preprocess(obs,acs)

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.x:inp[i:i+batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)

class GTDataset(object):
    def __init__(self,env,env_type):
        self.env = env
        self.env_type = env_type

        self.unwrapped = env
        while hasattr(self.unwrapped,'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self,agent,min_length):
        max_x_pos = -99999

        obs, actions, rewards = [self.env.reset()], [], []
        while True:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = self.env.step(action)

            if self.env_type == 'mujoco' and self.unwrapped.sim.data.qpos[0] > max_x_pos:
                max_x_pos = self.unwrapped.sim.data.qpos[0]

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                if len(obs) < min_length:
                    obs.pop()
                    obs.append(self.env.reset())
                else:
                    obs.pop()
                    break

        return (np.stack(obs,axis=0), np.array(actions), np.array(rewards)), max_x_pos

    def load_prebuilt(self,logdir):
        return False

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            traj, max_x_pos = self.gen_traj(agent,min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f max_x_pos: %f'%(agent.model_path,np.sum(traj[2]),max_x_pos))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape,self.trajs[2].shape)

    def sample(self,num_samples,steps=40,include_action=False):
        obs, actions, rewards = self.trajs

        D = []
        for _ in range(num_samples):
            x_ptr = np.random.randint(len(obs)-steps)
            y_ptr = np.random.randint(len(obs)-steps)

            if include_action:
                D.append((np.concatenate((obs[x_ptr:x_ptr+steps],actions[x_ptr:x_ptr+steps]),axis=1),
                         np.concatenate((obs[y_ptr:y_ptr+steps],actions[y_ptr:y_ptr+steps]),axis=1),
                         0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )
            else:
                D.append((obs[x_ptr:x_ptr+steps],
                          obs[y_ptr:y_ptr+steps],
                          0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )

        return D

class GTTrajLevelDataset(GTDataset):
    def __init__(self,env,env_type):
        super().__init__(env,env_type)

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent is given'

        trajs = []
        for agent_idx,agent in enumerate(tqdm(agents)):
            (obs,actions,rewards),_ = self.gen_traj(agent,min_length)
            trajs.append((agent_idx,obs,actions,rewards))

        self.trajs = trajs

        _idxes = np.argsort([np.sum(rewards) for _,_,_,rewards in self.trajs]) # rank 0 is the most bad demo.
        self.trajs_rank = np.empty_like(_idxes)
        self.trajs_rank[_idxes] = np.arange(len(_idxes))

    def sample(self,num_samples,steps=40,include_action=False):
        D = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            x_ptr = np.random.randint(len(x_traj[1])-steps)
            y_ptr = np.random.randint(len(y_traj[1])-steps)

            if include_action:
                D.append((np.concatenate((x_traj[1][x_ptr:x_ptr+steps],x_traj[2][x_ptr:x_ptr+steps]),axis=1),
                          np.concatenate((y_traj[1][y_ptr:y_ptr+steps],y_traj[2][y_ptr:y_ptr+steps]),axis=1),
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )
            else:
                D.append((x_traj[1][x_ptr:x_ptr+steps],
                          y_traj[1][y_ptr:y_ptr+steps],
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[3][x_ptr:x_ptr+steps]) > np.sum(y_traj[3][y_ptr:y_ptr+steps]) else 1)

        print('------------------')
        _,_,preference = zip(*D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)
        print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        print('------------------')

        return D

class GTTrajLevelNoStepsDataset(GTTrajLevelDataset):
    def __init__(self,env,env_type,max_steps):
        super().__init__(env,env_type)
        self.max_steps = max_steps

    def load_prebuilt(self,logdir):
        if os.path.exists(os.path.join(logdir,'prebuilt.npz')):
            f = np.load(os.path.join(logdir,'prebuilt.npz'))
            self.trajs = f['trajs']
            self.trajs_rank = f['trajs_rank']
            return True
        else:
            return False

    def prebuilt(self,agents,min_length,logdir=None):
        assert len(agents)>0, 'no agent is given'

        trajs = []
        for agent_idx,agent in enumerate(tqdm(agents)):
            agent_trajs = []
            while np.sum([len(obs) for obs,_,_ in agent_trajs])  < min_length:
                (obs,actions,rewards),_ = self.gen_traj(agent,-1)
                agent_trajs.append((obs,actions,rewards))
            trajs.append(agent_trajs)

            agent_reward = np.mean([np.sum(rewards) for _,_,rewards in agent_trajs])
            avg_len = np.mean([len(rewards) for _,_,rewards in agent_trajs])
            tqdm.write('model: %s eps len: %f avg reward: %f'%(agent.model_path,avg_len,agent_reward))

        agent_rewards = [np.mean([np.sum(rewards) for _,_,rewards in agent_trajs]) for agent_trajs in trajs]

        self.trajs = trajs

        _idxes = np.argsort(agent_rewards) # rank 0 is the most bad demo.
        self.trajs_rank = np.empty_like(_idxes)
        self.trajs_rank[_idxes] = np.arange(len(_idxes))

        if logdir is not None:
            np.savez(os.path.join(logdir,'prebuilt.npz'),trajs=self.trajs,trajs_rank=self.trajs_rank)

    def sample(self,num_samples,steps=None,include_action=False):
        assert steps == None

        D = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx][np.random.choice(len(self.trajs[x_idx]))]
            y_traj = self.trajs[y_idx][np.random.choice(len(self.trajs[y_idx]))]

            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[2][x_slice]) > np.sum(y_traj[2][y_slice]) else 1)

        print('------------------')
        _,_,preference = zip(*D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)
        print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        print('------------------')

        return D

class GTTrajLevelNoSteps_Noise_Dataset(GTTrajLevelNoStepsDataset):
    def __init__(self,env,env_type,max_steps,ranking_noise=0):
        super().__init__(env,env_type,max_steps)
        self.ranking_noise = ranking_noise

    def prebuilt(self,agents,min_length,logdir):
        super().prebuilt(agents,min_length,logdir)

        original_trajs_rank = self.trajs_rank.copy()
        for _ in range(self.ranking_noise):
            x = np.random.randint(len(self.trajs)-1)

            x_ptr = np.where(self.trajs_rank==x)
            y_ptr = np.where(self.trajs_rank==x+1)
            self.trajs_rank[x_ptr], self.trajs_rank[y_ptr] = x+1, x

        from itertools import combinations
        order_correctness = [
            (self.trajs_rank[x] < self.trajs_rank[y]) == (original_trajs_rank[x] < original_trajs_rank[y])
            for x,y in combinations(range(len(self.trajs)),2)]
        print('Total Order Correctness: %f'%(np.count_nonzero(order_correctness)/len(order_correctness)))

class GTTrajLevelNoSteps_N_Mix_Dataset(GTTrajLevelNoStepsDataset):
    def __init__(self,env,env_type,N,max_steps):
        super().__init__(env,env_type,max_steps)

        self.N = N
        self.max_steps = max_steps

    def sample(self,*kargs,**kwargs):
        return None

    def batch(self,batch_size,include_action):
        #self.trajs = trajs
        #self.trajs_rank = np.argsort([np.sum(rewards) for _,_,_,rewards in self.trajs]) # rank 0 is the most bad demo.
        xs = []
        ys = []

        for _ in range(batch_size):
            idxes = np.random.choice(len(self.trajs),2*self.N)

            ranks = self.trajs_rank[idxes]
            bad_idxes = [idxes[i] for i in np.argsort(ranks)[:self.N]]
            good_idxes = [idxes[i] for i in np.argsort(ranks)[self.N:]]

            def _pick_and_merge(idxes):
                inp = []
                for idx in idxes:
                    obs, acs, rewards = self.trajs[idx][np.random.choice(len(self.trajs[idx]))]

                    if len(obs) > self.max_steps:
                        ptr = np.random.randint(len(obs)-self.max_steps)
                        slc = slice(ptr,ptr+self.max_steps)
                    else:
                        slc = slice(len(obs))

                    if include_action:
                        inp.append(np.concatenate([obs[slc],acs[slc]],axis=1))
                    else:
                        inp.append(obs[slc])
                return np.concatenate(inp,axis=0)

            x = _pick_and_merge(bad_idxes)
            y = _pick_and_merge(good_idxes)

            xs.append(x)
            ys.append(y)

        x_split = np.array([len(x) for x in xs])
        y_split = np.array([len(y) for y in ys])
        xs = np.concatenate(xs,axis=0)
        ys = np.concatenate(ys,axis=0)

        return xs, ys, x_split, y_split, np.ones((batch_size,)).astype(np.int32)

class LearnerDataset(GTTrajLevelDataset):
    def __init__(self,env,env_type,min_margin):
        super().__init__(env,env_type)
        self.min_margin = min_margin

    def sample(self,num_samples,steps=40,include_action=False):
        D = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            x_ptr = np.random.randint(len(x_traj[1])-steps)
            y_ptr = np.random.randint(len(y_traj[1])-steps)

            if include_action:
                D.append((np.concatenate((x_traj[1][x_ptr:x_ptr+steps],x_traj[2][x_ptr:x_ptr+steps]),axis=1),
                         np.concatenate((y_traj[1][y_ptr:y_ptr+steps],y_traj[2][y_ptr:y_ptr+steps]),axis=1),
                         0 if x_traj[0] > y_traj[0] else 1)
                        )
            else:
                D.append((x_traj[1][x_ptr:x_ptr+steps],
                          y_traj[1][y_ptr:y_ptr+steps],
                         0 if x_traj[0] > y_traj[0] else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[3][x_ptr:x_ptr+steps]) > np.sum(y_traj[3][y_ptr:y_ptr+steps]) else 1)

        print('------------------')
        _,_,preference = zip(*D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)
        print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        print('------------------')

        return D


def train(args):
    # set random seed
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    # set log dir
    logdir = Path(args.log_dir)

    if logdir.exists() and 'temp' not in args.log_dir:
        c = input('log dir is already exist. continue to train a preference model? [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            return

    logdir.mkdir(parents=True,exist_ok=True)
    with open(str(logdir/'args.txt'),'w') as f:
        f.write( str(args) )
    logdir = str(logdir)

    # make env
    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
    elif args.env_type == 'atari':
        from baselines.common.atari_wrappers import make_atari, wrap_deepmind
        #TODO: episode_life True or False?
        env = wrap_deepmind(make_atari(args.env_id),episode_life=False,clip_rewards=False,frame_stack=True,scale=False)

    ob_shape = env.observation_space.shape
    ac_dims = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[-1]

    # make dataset (generate demonstration traj)
    if args.preference_type == 'gt':
        dataset = GTDataset(env,args.env_type)
    elif args.preference_type == 'gt_traj':
        dataset = GTTrajLevelDataset(env,args.env_type)
    elif args.preference_type == 'gt_traj_no_steps':
        dataset = GTTrajLevelNoStepsDataset(env,args.env_type,args.max_steps)
    elif args.preference_type == 'gt_traj_no_steps_noise':
        dataset = GTTrajLevelNoSteps_Noise_Dataset(env,args.env_type,args.max_steps,args.traj_noise)
    elif args.preference_type == 'gt_traj_no_steps_n_mix':
        dataset = GTTrajLevelNoSteps_N_Mix_Dataset(env,args.env_type,args.N,args.max_steps)
    elif args.preference_type == 'time':
        dataset = LearnerDataset(env,args.env_type,args.min_margin)
    else:
        assert False, 'specify prefernce type'

    loaded = dataset.load_prebuilt(logdir)

    if loaded == False:
        # load demonstrator
        train_agents = [RandomAgent(env.action_space)] if args.random_agent else []

        models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) <= args.max_chkpt])
        for path in models:
            agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
            train_agents.append(agent)

        dataset.prebuilt(train_agents,args.min_length,logdir)

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            if args.env_type == 'mujoco':
                net = MujocoNet(args.include_action,ob_shape[-1],ac_dims,num_layers=args.num_layers,embedding_dims=args.embedding_dims)
            elif args.env_type == 'atari':
                net = AtariNet(ob_shape,embedding_dims=args.embedding_dims)

            model = Model(net,batch_size=64)
            models.append(model)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        D = dataset.sample(args.D,args.steps,include_action=args.include_action)

        if D is None:
            model.train_with_dataset(dataset,64,include_action=args.include_action,debug=True)
        else:
            model.train(D,l2_reg=args.l2_reg,noise_level=args.noise,debug=True)

        model.saver.save(sess,logdir+'/model_%d.ckpt'%(i),write_meta_graph=False)

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    # Env Setting
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari', choices=['mujoco','atari'])
    # Demonstrator Setting
    parser.add_argument('--random_agent', action='store_true', help='whether to use default random agent')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=240, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    # Dataset setting
    parser.add_argument('--preference_type', help='gt or gt_traj or time or gt_traj_no_steps, gt_traj_no_steps_n_mix; if gt then preference will be given as a GT reward, otherwise, it is given as a time index')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--steps', default=None, type=int, help='length of snippets')
    parser.add_argument('--max_steps', default=None, type=int, help='length of max snippets (gt_traj_no_steps only)')
    parser.add_argument('--traj_noise', default=None, type=int, help='number of adjacent swaps (gt_traj_no_steps_noise only)')
    parser.add_argument('--N', default=10, type=int, help='number of trajactory mix (gt_traj_no_steps_n_mix only)')
    parser.add_argument('--min_margin', default=1, type=int, help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    # Network setting
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.1, type=float, help='noise level to add on training label')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', required=True, help='preference or preference_normalized')
    parser.add_argument('--num_timesteps', default=int(1e6), type=int)
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()

    if not args.eval :
        # Train a Preference Model
        #train(args)

        # Preferene based reward model analysis (draw figure)
        #import model_analysis
        #model_analysis.reward_analysis(args.log_dir)

        # Train an agent
        import os, subprocess, multiprocessing
        ncpu = multiprocessing.cpu_count()
        N.nvmlInit()
        ngpu = N.nvmlDeviceGetCount()

        openai_logdir = Path(os.path.abspath(os.path.join(args.log_dir,args.ppo_log_path)))
        if openai_logdir.exists():
            print('openai_logdir is already exist.')
            exit()

        template = 'python -m baselines.run --alg=ppo2 --env={env} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma}'
        kwargs = {
            "model_dir":os.path.abspath(args.log_dir),
            "ctrl_coeff":args.ctrl_coeff,
            "alive_bonus":args.alive_bonus
        }

        # Write down some log
        openai_logdir.mkdir(parents=True)
        with open(str(openai_logdir/'args.txt'),'w') as f:
            f.write( args.custom_reward + '/')
            f.write( str(kwargs) )

        cmd = template.format(
            env=args.env_id,
            nenv=ncpu//ngpu,
            num_timesteps=args.num_timesteps,
            save_interval=args.save_interval,
            custom_reward=args.custom_reward,
            gamma=args.gamma,
            kwargs=str(kwargs))

        procs = []
        for i in range(args.rl_runs):
            env = os.environ.copy()
            env["OPENAI_LOGDIR"] = str(openai_logdir/('run_%d'%i))
            if i == 0:
                env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
                p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
            else:
                env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
                p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)
            procs.append(p)

        for line in procs[0].stdout:
            print(line.decode(),end='')

        for p in procs[1:]:
            p.wait()

    else:
        import os
        from performance_checker import gen_traj as get_x_pos

        env = gym.make(args.env_id)

        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir,args.ppo_log_path)))
        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))

        print(str(agents_dir))
        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir/('run_%d'%i)/'checkpoints'/step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
                perfs += [
                    get_x_pos(env,agent) for _ in range(5)
                ]
                print('[%s-%d] %f %f'%(step,i,np.mean(perfs[-5:]),np.std(perfs[-5:])))

            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)))

            #break
