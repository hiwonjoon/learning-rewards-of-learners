import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
from siamese_ranker import PPO2Agent, RandomAgent
from behavior_cloning import Dataset
from tf_commons.ops import *

class InverseDynamics(object):
    def __init__(self,ob_shape,ac_dim,embed_size=128):
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            self.inp = tf.placeholder(tf.float32,[None,ob_shape[0],ob_shape[1],ob_shape[2]])
            self.l = tf.placeholder(tf.int32,[None])
            self.l2_reg = tf.placeholder(tf.float32,[])

            with tf.variable_scope('weights') as param_scope:
                self.conv1 = Conv2d('conv1',4,32,k_h=8,k_w=8,d_h=4,d_w=4,data_format='NHWC',padding='VALID')
                self.conv2 = Conv2d('conv2',32,64,k_h=4,k_w=4,d_h=2,d_w=2,data_format='NHWC',padding='VALID')
                self.conv3 = Conv2d('conv3',64,64,k_h=3,k_w=3,d_h=2,d_w=2,data_format='NHWC',padding='VALID')
                self.fc1 = Linear('fc1',64*4*4, embed_size)
                self.fc2 = Linear('fc2',embed_size, ac_dim)

            self.param_scope = param_scope

            # build graph
            def _build(x):
                _ = tf.nn.relu(self.conv1(x))
                _ = tf.nn.relu(self.conv2(_))
                _ = tf.nn.relu(self.conv3(_))
                _ = tf.nn.relu(self.fc1(_))
                logits = self.fc2(_)
                return logits

            B = tf.shape(self.inp)[0]

            self.logits = _build(self.inp)
            self.pred = tf.cast(tf.argmax(self.logits,axis=1),tf.int32)

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.l),axis=0)
            self.acc = tf.cast(tf.count_nonzero(tf.cast(tf.equal(self.pred,self.l),tf.int32)),tf.float32) / tf.cast(B,tf.float32)

            weight_decay = \
                tf.reduce_sum(self.conv1.w**2) + \
                tf.reduce_sum(self.conv2.w**2) + \
                tf.reduce_sum(self.conv3.w**2) + \
                tf.reduce_sum(self.fc1.w**2) + \
                tf.reduce_sum(self.fc2.w**2)

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

    def train(self,D,batch_size=64,iter=20000,l2_reg=0.001,debug=False):
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
                batch.append((obs[i]-obs[i+1],acs[i]))
                #batch.append((np.concatenate([obs[i],obs[i+1]],axis=2),acs[i]))

            b_ob,b_ac = zip(*batch)
            b_ob,b_ac = np.array(b_ob),np.array(b_ac)

            return b_ob,b_ac

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_ob,b_ac = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                    self.inp:b_ob,
                    self.l:b_ac,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_ob,b_ac = _batch(valid_idxes)
                    valid_loss, valid_acc = sess.run([self.loss,self.acc],feed_dict={
                        self.inp:b_ob,
                        self.l:b_ac,
                        self.l2_reg:l2_reg,
                    })
                    tqdm.write(('loss: %f acc: %f (l2_loss: %f), valid_loss: %f valid_acc: %f'%(loss,acc,l2_loss,valid_loss,valid_acc)))

    def infer_action(self, b_obs, batch_size=64):
        sess = self.sess

        with self.graph.as_default():
            ac = []
            for s in range(0,len(b_obs)-1,batch_size):
                logits = sess.run(self.logits,feed_dict={
                    self.inp:b_obs[s:min(s+batch_size,len(b_obs)-1)]-b_obs[s+1:min(s+1+batch_size,len(b_obs))]})
                    #self.inp:np.concatenate([b_obs[s:min(s+batch_size,len(b_obs)-1)],b_obs[s+1:min(s+1+batch_size,len(b_obs))]],axis=3)})
                ac.append(np.argmax(logits,axis=1))

        return np.concatenate(ac,axis=0)

    def save(self,path):
        with self.graph.as_default():
            self.saver.save(self.sess,path,write_meta_graph=False)

    def load(self,path):
        with self.graph.as_default():
            self.saver.restore(self.sess,path)


class Agent(object):
    def __init__(self, env, env_type, nenv=4, batch_size=64, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model

        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0})

        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env,'cnn')
                    target_action = tf.placeholder(tf.int32,[batch_size])
                elif env_type == 'mujoco':
                    policy = build_policy(env,'mlp')
                    target_action = tf.placeholder(tf.float32,[batch_size,ac_space.shape[0]])
                else:
                    assert False,' not supported env_type'

                make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenv, nbatch_train=batch_size,
                                nsteps=1, ent_coef=0., vf_coef=0.,max_grad_norm=0.)
                self.model = make_model()

                self.inp = self.model.train_model.X # This is also placeholder
                self.target_action = target_action

                self.ac_logits = self.model.train_model.pi

                if env_type == 'atari':
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.ac_logits,
                                    labels=self.target_action)
                elif env_type == 'mujoco':
                    loss = tf.reduce_sum((self.ac_logits-self.l)**2,axis=1)

                self.loss = tf.reduce_mean(loss,axis=0)

                policy_optim = tf.train.AdamOptimizer(1e-4)
                policy_params = tf.trainable_variables('ppo2_model/pi')
                self.update_op = policy_optim.minimize(self.loss,var_list=policy_params)

                # Value Fn Optimization
                self.R = R = tf.placeholder(tf.float32, [None])
                self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
                self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

                vpred = self.model.train_model.vf
                vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE)
                # Unclipped value
                vf_losses1 = tf.square(vpred - R)
                # Clipped value
                vf_losses2 = tf.square(vpredclipped - R)

                self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                value_optim = tf.train.AdamOptimizer(1e-4)
                value_params = tf.trainable_variables('ppo2_model/vf')
                self.value_update_op = value_optim.minimize(self.vf_loss,var_list=value_params)

                ################ Miscellaneous
                self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

            self.sess.run(self.init_op)

    def train(self,D,batch_size=64,iter=20000,l2_reg=0.001,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs))
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append((obs[i],acs[i]))

            b_ob,b_ac = zip(*batch)
            b_ob,b_ac = np.array(b_ob),np.array(b_ac)

            return b_ob,b_ac

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_ob,b_ac = _batch(train_idxes)

            with self.graph.as_default():
                loss,_ = sess.run([self.loss,self.update_op],feed_dict={
                    self.inp:b_ob,
                    self.target_action:b_ac,
                })


            if debug:
                if it % 100 == 0 or it < 10:
                    b_ob,b_ac = _batch(valid_idxes)
                    valid_loss= sess.run(self.loss,feed_dict={
                        self.inp:b_ob,
                        self.target_action:b_ac,
                    })
                    tqdm.write(('loss: %f valid_loss: %f'%(loss,valid_loss)))

    def train_value(self,env,env_type,nupdates,minibatch_size=64):
        from baselines.ppo2.runner import Runner
        import baselines.ppo2.defaults as defaults

        if env_type == 'mujoco':
            params = defaults.mujoco()
        elif env_type == 'atari':
            params = defaults.atari()
        else:
            assert False

        runner = Runner(env=env, model=self.model, nsteps=params['nsteps'], gamma=params['gamma'], lam=params['lam'])

        for update in tqdm(range(1, nupdates+1),dynamic_ncols=True):
            frac = 1.0 - (update - 1.0) / nupdates
            cliprangenow = params['cliprange'](frac)
            # Get minibatch
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

            length = len(obs)
            losses = []
            for _ in range(params['noptepochs']):
                inds = np.random.permutation(length)

                for s in range(0, length, minibatch_size):
                    mbinds = inds[s:s+minibatch_size]

                    with self.graph.as_default():
                        loss,_ = self.sess.run([self.vf_loss,self.value_update_op],feed_dict={
                            self.inp: obs[mbinds],
                            self.R: returns[mbinds],
                            self.OLDVPRED: values[mbinds],
                            self.CLIPRANGE: cliprangenow
                        })
                        losses.append(loss)
            tqdm.write(('loss: %f')%(np.mean(losses)))

    def save(self,path):
        with self.graph.as_default():
            self.model.save(path)


def train(args):
    logdir = Path(args.logdir)
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

    logdir = str(logdir)

    from baselines.common.cmd_util import make_vec_env
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    env = make_vec_env(args.env_id, args.env_type, 1, 0,
                        wrapper_kwargs={
                            'clip_rewards':False,
                            'episode_life':False,
                        })
    if args.env_type == 'atari':
        env = VecFrameStack(env, 4)
    elif args.env_type == 'mujoco':
        pass
    else:
        assert False

    ### Train Inverse Dynamics Model
    if args.BCO:
        inv_model = InverseDynamics(env.observation_space.shape,env.action_space.n)
        if args.pretrained_inv:
            inv_model.load(args.pretrained_inv)
        else:
            dataset = Dataset(env)
            dataset.prebuilt([RandomAgent(env.observation_space,env.action_space)],args.inv_min_length)

            inv_model.train(dataset.trajs,iter=args.inv_min_length,debug=True)
            inv_model.save(logdir+'/inv_model.ckpt')

    ### Train Policy with Behavior Cloning
    agents = []
    if args.best_only:
        agent = PPO2Agent(env,args.env_type,str(Path(args.learners_path)/('%05d'%600)),stochastic=args.stochastic)
        agents.append(agent)
    else:
        for i in range(50,601,50):
            agent = PPO2Agent(env,args.env_type,str(Path(args.learners_path)/('%05d'%i)),stochastic=args.stochastic)
            agents.append(agent)

    dataset = Dataset(env)
    dataset.prebuilt(agents,args.min_length)

    if args.BCO:
        obs,acs,_ = dataset.trajs
        inf_acs = inv_model.infer_action(obs)

        with open(logdir+'/inv_dy_acc.txt','w') as f:
            f.write('inv_dynamics accuracy: %f'%(np.count_nonzero(acs[:-1] == inf_acs)/len(inf_acs)))
        D = (obs[:-1],inf_acs,None)
    else:
        D = dataset.trajs

    learner = Agent(env, args.env_type)
    learner.train(D,iter=args.iter,debug=True)

    learner.save(logdir+'/bc_model_policy_only')

    ### Train Value
    if args.env_type == 'atari':
        env = make_vec_env(args.env_id, args.env_type, args.nenv, 0,
                            wrapper_kwargs={
                                'clip_rewards':False,
                                'episode_life':False,
                            })
        env = VecFrameStack(env, 4)
    elif args.env_type == 'mujoco':
        assert False, 'Not Implemented'
    else:
        assert False

    learner.train_value(env,args.env_type,nupdates=args.nupdates)

    learner.save(logdir+'/bc_model_full')

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', help='Select the environment to run', required=True)
    parser.add_argument('--env_type', help='mujoco or atari', required=True)
    parser.add_argument('--logdir', default='./_log_temp/', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--BCO', action='store_true', help='BCO or BC')
    ### For BCO
    parser.add_argument('--pretrained_inv', default=None)
    parser.add_argument('--inv_min_length', default=100000,type=int, help='minimum length of trajectory to train inv_dynamis')
    parser.add_argument('--inv_iter', default=20000,type=int, help='# sgd steps for inverse dynamcis learning')
    ### For BC
    parser.add_argument('--learners_path', help='path of learning agents', required=True)
    parser.add_argument('--best_only', action='store_true', help='whether behavior clone only the best only or mix up all')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--min_length', default=-1,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--iter', default=20000,type=int, help='number of SGD steps')
    #### For Value Training
    parser.add_argument('--nenv', default=4, type=int)
    parser.add_argument('--nupdates', default=1000, type=int)
    #### For eval
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    #parser.add_argument('--max_len', default=1000, type=int)
    #parser.add_argument('--num_trajs', default=10, type=int, help='path of learning agents')
    #parser.add_argument('--render', action='store_true')
    #parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()

    if not args.eval :
        train(args)
    else:
        eval(args)

#def eval(args):
#    env = gym.make(args.env_id)
#
#    logdir = str(Path(args.logbase_path) / args.env_id)
#    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0])
#
#    ### Initialize Parameters
#    init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#    # Training configuration
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    sess = tf.InteractiveSession()
#
#    sess.run(init_op)
#    policy.saver.restore(sess,logdir+'/model.ckpt')
#
#    from performance_checker import gen_traj
#    from gym.wrappers import Monitor
#    perfs = []
#    for j in range(args.num_trajs):
#        if j == 0 and args.video_record:
#            wrapped = Monitor(env, './video/',force=True)
#        else:
#            wrapped = env
#
#        perfs.append(gen_traj(wrapped,policy,args.render,args.max_len))
#    print(logdir, ',', np.mean(perfs), ',', np.std(perfs))


