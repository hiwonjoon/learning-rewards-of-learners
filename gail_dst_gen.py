from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import gym

from siamese_ranker import PPO2Agent

def gen_traj(env,agent,min_length):
    obs, actions, rewards = [env.reset()], [], []
    while True:
        action = agent.act(obs[-1], None, None)
        ob, reward, done, _ = env.step(action)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)

        if done:
            if len(obs) < min_length:
                obs.pop()
                obs.append(env.reset())
            else:
                obs.pop()
                break

    return (np.stack(obs,axis=0), np.array(actions), np.array(rewards))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # Env Setting
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='mujoco', help='mujoco or atari', choices=['mujoco'])
    # Demonstrator Setting
    parser.add_argument('--learners_path', required=True, help='path of learning agents')
    parser.add_argument('--train_chkpt', default='240', help='decide upto what learner stage you want to give')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    # Num Trajs per each agent
    parser.add_argument('--num_trajs', default=1, type=int) # Generate 24 to compare with GAIL
    parser.add_argument('--min_length', default=1000, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    train_chkpt = eval(args.train_chkpt)
    if type(train_chkpt) == int:
        train_chkpt = list(range(train_chkpt+1))
    else:
        train_chkpt = list(train_chkpt)

    env = gym.make(args.env_id)

    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) in train_chkpt])
    train_agents = []
    for path in models:
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
        train_agents.append(agent)

    obs_list = []
    acs_list = []
    rews_list  = []
    ep_rets = []
    for agent in train_agents:
        for _ in range(args.num_trajs):
            obs,acs,rews = gen_traj(env,agent,args.min_length)

            obs_list.append(obs)
            acs_list.append(acs)
            rews_list.append(rews)
            ep_rets.append(np.sum(rews))

    import string
    filename = '%s_%s_%d.npz'%(args.env_id,args.train_chkpt,args.num_trajs)
    filename = ''.join(c for c in filename if c in "-_.%s%s" % (string.ascii_letters, string.digits))

    np.savez(
        filename,
        obs=obs_list,
        acs=acs_list,
        rews=rews_list,
        ep_rets=np.array(ep_rets))
