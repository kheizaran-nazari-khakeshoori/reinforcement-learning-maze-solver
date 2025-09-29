"""Compare."""
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent, SarsaAgent
from train import run_episode
def run_compare(episodes=100):
    env_q=MazeEnv(size=4); env_s=MazeEnv(size=4)
    q=QLearningAgent(env_q.observation_space.n, env_q.action_space.n)
    s=SarsaAgent(env_s.observation_space.n, env_s.action_space.n)
    q_rewards=[]; s_rewards=[]
    eps=1.0
    for _ in range(episodes):
        r,_=run_episode(env_q,q,eps)
        q_rewards.append(r)
        # sarsa manual
        state,_=env_s.reset(); total=0; done=False; a=s.act(state,eps)
        while not done:
            ns,rew,term,trunc,_=env_s.step(a)
            done=term or trunc
            na=s.act(ns,eps) if not done else 0
            s.update(state,a,rew,ns,na,done)
            state=ns; a=na; total+=rew
        s_rewards.append(total)
        eps=max(0.05, eps*0.995)
    return q_rewards, s_rewards
