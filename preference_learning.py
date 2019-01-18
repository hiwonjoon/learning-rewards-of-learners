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

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()[None]

class Model(object):
    def __init__(self,include_action,ob_dim,ac_dim,embedding_dims=256,steps=40):
        self.include_action = include_action
        in_dims = ob_dim+ac_dim if include_action else ob_dim

        self.inp = tf.placeholder(tf.float32,[None,in_dims])
        self.x = tf.placeholder(tf.float32,[None,steps,in_dims])
        self.y = tf.placeholder(tf.float32,[None,steps,in_dims])
        self.l = tf.placeholder(tf.int32,[None]) # [0 when x is better 1 when y is better]
        self.l2_reg = tf.placeholder(tf.float32,[]) # [0 when x is better 1 when y is better]


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

        self.r = _reward(self.inp)

        self.r_x = tf.reshape(_reward(tf.reshape(self.x,[-1,in_dims])),[-1,steps])
        self.sum_r_x = tf.reduce_sum(self.r_x,axis=1)

        self.r_y = tf.reshape(_reward(tf.reshape(self.y,[-1,in_dims])),[-1,steps])
        self.sum_r_y = tf.reduce_sum(self.r_y,axis=1)

        logits = tf.stack([self.sum_r_x,self.sum_r_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        weight_decay = tf.reduce_sum(self.fc1.w**2) + tf.reduce_sum(self.fc2.w**2) + tf.reduce_sum(self.fc3.w**2)
        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.sum_r_y,self.sum_r_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size=64,iter=10000,l2_reg=0.01,noise_level=0.1,debug=False):
        """
        Training will be early-terminate when validation accuracy becomes large enough..?

        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
            while
                sigma^{1,2}: shape of [steps,in_dims]
                mu : 0 or 1
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D)*0.8)]
        valid_idxes = idxes[int(len(D)*0.8):]

        def _batch(idx_list,add_noise):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append(D[i])

            b_x,b_y,b_l = zip(*batch)
            b_x,b_y,b_l = np.array(b_x),np.array(b_y),np.array(b_l)

            if add_noise:
                b_l = (b_l + np.random.binomial(1,noise_level,batch_size)) % 2 #Flip it with probability 0.1

            return b_x,b_y,b_l

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_x,b_y,b_l = _batch(train_idxes,add_noise=True)
            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            b_x,b_y,b_l = _batch(valid_idxes,add_noise=False)
            valid_acc = sess.run(self.acc,feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.l:b_l
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))

            #if valid_acc >= 0.95:
            #    print('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc))
            #    print('early termination@%08d'%it)
            #    break

    def eval(self,D,batch_size=64):
        sess = tf.get_default_session()

        b_x,b_y,b_l = zip(*D)
        b_x,b_y,b_l = np.array(b_x),np.array(b_y),np.array(b_l)

        b_r_x, b_acc = [], 0.

        for i in range(0,len(b_x),batch_size):
            sum_r_x, acc = sess.run([self.sum_r_x,self.acc],feed_dict={
                self.x:b_x[i:i+batch_size],
                self.y:b_y[i:i+batch_size],
                self.l:b_l[i:i+batch_size]
            })

            b_r_x.append(sum_r_x)
            b_acc += len(sum_r_x)*acc

        return np.concatenate(b_r_x,axis=0), b_acc/len(b_x)

    def get_reward(self,obs,acs,batch_size=1024):
        sess = tf.get_default_session()

        if self.include_action:
            inp = np.concatenate((obs,acs),axis=1)
        else:
            inp = obs

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.inp:inp[i:i+batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)

class GTDataset(object):
    def __init__(self,env):
        self.env = env
        self.unwrapped = env
        while hasattr(self.unwrapped,'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self,agent,min_length):
        max_x_pos = -99999

        obs, actions, rewards = [self.env.reset()], [], []
        while True:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = self.env.step(action)
            if self.unwrapped.sim.data.qpos[0] > max_x_pos:
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

        return (np.stack(obs,axis=0), np.concatenate(actions,axis=0), np.array(rewards)), max_x_pos

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
    def __init__(self,env):
        super().__init__(env)

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent is given'

        trajs = []
        for agent_idx,agent in enumerate(tqdm(agents)):
            (obs,actions,rewards),_ = self.gen_traj(agent,min_length)
            trajs.append((agent_idx,obs,actions,rewards))

        self.trajs = trajs
        self.trajs_rank = np.argsort([np.sum(rewards) for _,_,_,rewards in self.trajs]) # rank 0 is the most bad demo.

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


class LearnerDataset(GTTrajLevelDataset):
    def __init__(self,env,min_margin):
        super().__init__(env)
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
    logdir = Path(args.log_dir)

    if logdir.exists() :
        c = input('log dir is already exist. continue to train a preference model? [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            return

    logdir.mkdir(parents=True)
    with open(str(logdir/'args.txt'),'w') as f:
        f.write( str(args) )

    logdir = str(logdir)
    env = gym.make(args.env_id)

    train_agents = [RandomAgent(env.action_space)] if args.random_agent else []

    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) <= args.max_chkpt])
    for path in models:
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
        train_agents.append(agent)

    if args.preference_type == 'gt':
        dataset = GTDataset(env)
    elif args.preference_type == 'gt_traj':
        dataset = GTTrajLevelDataset(env)
    elif args.preference_type == 'time':
        dataset = LearnerDataset(env,args.min_margin)
    else:
        assert False, 'specify prefernce type'

    dataset.prebuilt(train_agents,args.min_length)

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            models.append(Model(args.include_action,env.observation_space.shape[0],env.action_space.shape[0],steps=args.steps))

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
        model.train(D,l2_reg=args.l2_reg,noise_level=args.noise,debug=True)

        model.saver.save(sess,logdir+'/model_%d.ckpt'%(i),write_meta_graph=False)

def eval(args):
    logdir = str(Path(args.logbase_path) / args.env_id)

    env = gym.make(args.env_id)

    valid_agents = []
    models = sorted(Path(args.learners_path).glob('?????'))
    for path in models:
        if path.name > args.max_chkpt:
            continue
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
        valid_agents.append(agent)

    test_agents = []
    for i,path in enumerate(models):
        if i % 10 == 0:
            agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
            test_agents.append(agent)

    gt_dataset= GTDataset(env)
    gt_dataset.prebuilt(valid_agents,-1)

    gt_dataset_test = GTDataset(env)
    gt_dataset_test.prebuilt(test_agents,-1)

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            models.append(Model(args.include_action,env.observation_space.shape[0],env.action_space.shape[0],steps=args.steps))

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        model.saver.restore(sess,logdir+'/model_%d.ckpt'%(i))

        print('model %d'%i)
        obs, acs, r = gt_dataset.trajs
        r_hat = model.get_reward(obs, acs)

        obs, acs, r_test = gt_dataset_test.trajs
        r_hat_test = model.get_reward(obs, acs)

        fig,axes = plt.subplots(1,2)
        axes[0].plot(r,r_hat,'o')
        axes[1].plot(r_test,r_hat_test,'o')
        fig.savefig('model_%d.png'%i)
        imgcat(fig)
        plt.close(fig)

        np.savez('model_%d.npz'%i,r=r,r_hat=r_hat,r_test=r_test,r_hat_test=r_hat_test)


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=240, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--steps', default=40, type=int, help='length of snippets')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.1, type=float, help='noise level to add on training label')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--preference_type', help='gt or gt_traj or time; if gt then preference will be given as a GT reward, otherwise, it is given as a time index')
    parser.add_argument('--min_margin', default=1, type=int, help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--random_agent', action='store_true', help='whether to use default random agent')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', required=True, help='preference or preference_normalized')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    args = parser.parse_args()

    if not args.eval :
        # Train a Preference Model
        train(args)

        # Train an agent
        import os, subprocess
        openai_logdir = Path(os.path.abspath(os.path.join(args.log_dir,args.ppo_log_path)))
        if openai_logdir.exists():
            print('openai_logdir is already exist.')
            exit()

        template = 'python -m baselines.run --alg=ppo2 --env={env} --num_timesteps=1e6 --save_interval=10 --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}"'
        kwargs = {
            "num_models":args.num_models,
            "include_action":args.include_action,
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
            custom_reward=args.custom_reward,
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
        eval(args)
