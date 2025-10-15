"""Evaluate."""
from maze_solver.env import MazeEnv
def evaluate(agent, env, episodes=10, max_steps=50):
    succ=0; total_steps=0
    for _ in range(episodes):
        s,_=env.reset(); done=False; steps=0
        while not done and steps<max_steps:
            a=agent.act(s, 0.0)
            ns,_,term,trunc,_=env.step(a)
            done=term or trunc
            s=ns; steps+=1
            if done and env.agent_pos==env.goal:
                succ+=1
        total_steps+=steps
    return {"success":succ/episodes, "avg_steps":total_steps/episodes}

def quick_eval(agent, env):
    return evaluate(agent, env, episodes=5)

def success_rate(agent, env):
    return evaluate(agent, env)["success"]
