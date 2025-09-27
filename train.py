"""Train."""
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
def run_episode(env, agent, eps=0.1):
    s,_=env.reset(); total=0; steps=0; done=False
    while not done:
        a=agent.act(s, eps)
        ns,r,term,trunc,_=env.step(a)
        done=term or trunc
        try:
            agent.update(s,a,r,ns,done)
        except TypeError:
            agent.update(s,a,r,ns,a,done)
        s=ns; total+=r; steps+=1
    return total, steps

def train(agent, env, episodes=100, eps=1.0, decay=0.995, min_eps=0.05):
    cur=eps; rewards=[]; steps=[]
    for ep in range(episodes):
        r,s=run_episode(env, agent, cur)
        rewards.append(r); steps.append(s)
        cur=max(min_eps, cur*decay)
    return rewards, steps

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    args=p.parse_args()
    env=MazeEnv(size=5)
    agent=QLearningAgent(env.observation_space.n, env.action_space.n)
    train(agent, env, episodes=args.episodes)
