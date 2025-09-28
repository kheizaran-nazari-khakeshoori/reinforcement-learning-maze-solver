"""Evaluate."""
from maze_solver.env import MazeEnv
def evaluate(agent, env, episodes=10):
    succ=0
    for _ in range(episodes):
        s,_=env.reset(); done=False
        while not done:
            a=agent.act(s, 0.0)
            ns,_,term,trunc,_=env.step(a)
            done=term or trunc
            s=ns
            if done and env.agent_pos==env.goal:
                succ+=1
                break
    return succ/episodes
