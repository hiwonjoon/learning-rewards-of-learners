import os
import argparse
import gym
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
from pathlib import Path
from matplotlib import pyplot as plt
from imgcat import imgcat

from siamese_ranker import PPO2Agent
from preference_learning import Model, MujocoNet, AtariNet

# Matplotlib setting
rcParams = {'legend.fontsize': 'xx-large',
          'figure.figsize': (6, 5),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large'}
matplotlib.rcParams.update(rcParams)

def linear_model_analysis(args):
    # make env
    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
    elif args.env_type == 'atari':
        from baselines.common.atari_wrappers import make_atari, wrap_deepmind
        #TODO: episode_life True or False?
        env = wrap_deepmind(make_atari(args.env_id),episode_life=False,clip_rewards=False,frame_stack=True,scale=False)

    train_agents = []
    test_agents = []
    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) <= args.max_chkpt])
    for path in models:
        agent = PPO2Agent(env,'mujoco',str(path),stochastic=args.stochastic)

        if int(path.name) <= args.model_max_chkpt:
            train_agents.append(agent)
        else:
            test_agents.append(agent)

    from preference_learning import GTTrajLevelNoStepsDataset as Dataset
    dataset = Dataset(env,1000)
    dataset.prebuilt(train_agents+test_agents,1000)

    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    with tf.variable_scope('model_0'):
        model = Model(args.include_action,env.observation_space.shape[0],env.action_space.shape[0],steps=None,num_layers=0,embedding_dims=args.embedding_dims)
    model.saver.restore(sess,args.model_path)

    x = []
    yy = []
    seen = []
    for i,(agent_trajs,rank) in enumerate(zip(dataset.trajs,dataset.trajs_rank)):
        for obs,_,_ in agent_trajs:
            obs,_,_ = agent_trajs[0]
            yy.append(np.sum(obs,axis=0))
            x.append(rank)
            seen.append(True if i < len(train_agents) else False)
    x = np.array(x)
    yy = np.stack(yy,axis=1) # [#features, #agents]
    seen_ptr = np.max(np.where(seen))

    w_scale = np.std(yy[:,:seen_ptr],axis=1)

    w = sess.run(model.fcs[-1].w)[:,0]
    for idx in np.argsort(np.abs(w*w_scale)):
        print(idx,w[idx],w[idx]*w_scale[idx])

    for idx in np.argsort(np.abs(w*w_scale))[::-1]:
        fig,ax = plt.subplots()
        ax.set_title('idx: %d importance(scaled): %f'%(idx,w_scale[idx]*w[idx]))
        ax.scatter(x=x[:seen_ptr],y=yy[idx,:seen_ptr],color='blue')
        ax.scatter(x=x[seen_ptr:],y=yy[idx,seen_ptr:],color='red')
        imgcat(fig)

        input()
        plt.close(fig)

def reward_analysis(model_path):
    from argparse import Namespace
    with open(str(Path(model_path)/'args.txt')) as f:
        args = eval(f.read())

    # make env
    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
    elif args.env_type == 'atari':
        from baselines.common.atari_wrappers import make_atari, wrap_deepmind
        #TODO: episode_life True or False?
        env = wrap_deepmind(make_atari(args.env_id),episode_life=False,clip_rewards=False,frame_stack=True,scale=False)

    ob_shape = env.observation_space.shape
    ac_dims = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[-1]

    train_agents = []
    test_agents = []
    models = sorted([p for p in Path(args.learners_path).glob('?????')]) # if int(p.name) <= args.max_chkpt])
    for path in models:
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)

        if int(path.name) <= args.max_chkpt:
            train_agents.append(agent)
        else:
            test_agents.append(agent)

    from preference_learning import GTTrajLevelNoStepsDataset as Dataset
    dataset = Dataset(env,args.env_type,1000)
    dataset.prebuilt(train_agents+test_agents,1000)

    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            if args.env_type == 'mujoco':
                net = MujocoNet(args.include_action,ob_shape[-1],ac_dims,num_layers=args.num_layers,embedding_dims=args.embedding_dims)
            elif args.env_type == 'atari':
                net = AtariNet(ob_shape,embedding_dims=args.embedding_dims)

            model = Model(net,batch_size=1)
            model.saver.restore(sess,os.path.join(model_path,'model_%d.ckpt'%i))

            models.append(model)

    rank_acc_r_pts = []
    acc_r_pts = []
    r_pts = []
    for agent_idx,(agent_trajs,rank) in enumerate(zip(dataset.trajs,dataset.trajs_rank)):
        for obs,acs,rs in agent_trajs:
            rs_hats = []
            for i,model in enumerate(models):
                rs_hats.append(model.get_reward(obs, acs))
            rs_hats = np.stack(rs_hats,axis=0) # [#models, #length of the traj]

            rs_hat = np.mean(rs_hats,axis=0) # [# length of the traj]
            acc_r_hat = np.sum(rs_hat,axis=0)

            # GT ACC R
            acc_r = np.sum(rs)

            r_pts += [(r,r_hat,agent_idx < len(train_agents)) for r,r_hat in zip(rs,rs_hat)]
            acc_r_pts.append((acc_r,acc_r_hat,agent_idx < len(train_agents)))
            rank_acc_r_pts.append((rank,acc_r_hat,agent_idx < len(train_agents)))

    np.savez(model_path+'/reward_analysis.npz',rank_acc_r_pts=rank_acc_r_pts,acc_r_pts=acc_r_pts)

    def convert_range(x,minimum, maximum,a,b):
        return (x - minimum)/(maximum - minimum) * (b - a) + a\

    def draw(gt_returns, pred_returns, seen):
        seen_ptr = np.max(np.where(seen))

        gt_max,gt_min = max(gt_returns),min(gt_returns)
        pred_max,pred_min = max(pred_returns),min(pred_returns)
        max_observed = max(pred_returns[:seen_ptr])

        # Draw P
        fig,ax = plt.subplots()

        ax.plot(gt_returns[:seen_ptr], [convert_range(p,pred_max,pred_min,gt_max,gt_min) for p in pred_returns[:seen_ptr]], 'o', color='red')
        ax.plot(gt_returns[seen_ptr:], [convert_range(p,pred_max,pred_min,gt_max,gt_min) for p in pred_returns[seen_ptr:]], 'o', color='blue')
        ax.plot([gt_min-5,gt_max+5],[gt_min-5,gt_max+5],'g--')
        ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)

        ax.axis([gt_min-5,gt_max+5,gt_min-5,gt_max+5])
        ax.set_xlabel("Ground Truth Returns")
        ax.set_ylabel("Predicted Returns (normalized)")
        fig.tight_layout()
        fig.savefig(model_path+'/acc_r_vs_acc_r_hat.png',dpi=400)

        plt.close(fig)

    draw(*[np.array(e) for e in zip(*acc_r_pts)])

    """
    fig,ax = plt.subplots()
    x,y,seen = (np.array(e) for e in zip(*rank_acc_r_pts))
    seen_ptr = np.max(np.where(seen))
    ax.scatter(x=x[:seen_ptr],y=y[:seen_ptr],color='blue')
    ax.scatter(x=x[seen_ptr:],y=y[seen_ptr:],color='red')
    fig.savefig(args.model_path+'/rank_vs_acc_r_hat.png')
    #imgcat(fig)
    plt.close(fig)

    fig,ax = plt.subplots()
    x,y,seen = (np.array(e) for e in zip(*acc_r_pts))
    seen_ptr = np.max(np.where(seen))
    ax.scatter(x=x[:seen_ptr],y=y[:seen_ptr],color='blue')
    ax.scatter(x=x[seen_ptr:],y=y[seen_ptr:],color='red')
    fig.savefig(args.model_path+'/acc_r_vs_acc_r_hat.png')
    #imgcat(fig)
    plt.close(fig)

    fig,ax = plt.subplots()
    x,y,seen = (np.array(e) for e in zip(*r_pts))
    seen_ptr = np.max(np.where(seen))
    ax.scatter(x=x[:seen_ptr],y=y[:seen_ptr],color='blue')
    ax.scatter(x=x[seen_ptr:],y=y[seen_ptr:],color='red')
    fig.savefig(args.model_path+'/scatter_r_vs_r_hat.png')
    #imgcat(fig)
    plt.close(fig)
    """


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model_path', required=True)

    args = parser.parse_args()

    #linear_model_analysis(args)
    reward_analysis(args.model_path)
