"""Train."""
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
def run_episode(env, agent, eps=0.1):
    s,_=env.reset(); total=0; done=False
    while not done:
        a=agent.act(s, eps)
        ns,r,term,trunc,_=env.step(a)
        done=term or trunc
        try:
            agent.update(s,a,r,ns,done)
        except TypeError:
            agent.update(s,a,r,ns,a,done)
        s=ns; total+=r
    return total
