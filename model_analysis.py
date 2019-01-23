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
from preference_learning import Model

def linear_model_analysis(args):
    env = gym.make(args.env_id)

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

def reward_analysis(args):
    env = gym.make(args.env_id)

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

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            models.append(
                Model(args.include_action,env.observation_space.shape[0],env.action_space.shape[0],steps=None,num_layers=args.num_layers,embedding_dims=args.embedding_dims))
        models[-1].saver.restore(sess,args.model_path+'/model_%d.ckpt'%i)

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

    np.savez(args.model_path+'/reward_analysis.npz',rank_acc_r_pts=rank_acc_r_pts,acc_r_pts=acc_r_pts)

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


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--num_layers', default=2, type=int, help='embedding dims')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=5, type=int, help='number of models to ensemble')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_max_chkpt', default=240, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=240, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')

    args = parser.parse_args()

    #linear_model_analysis(args)
    reward_analysis(args)
