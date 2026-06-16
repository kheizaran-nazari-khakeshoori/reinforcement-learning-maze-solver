"""Evaluation with seed support."""

from typing import Optional
from maze_solver.env import MazeEnv

def evaluate(agent, env, episodes: int = 10, max_steps: int = 50, seed: Optional[int] = None) -> dict:
    succ=0; total_steps=0
    for ep in range(episodes):
        ep_seed = seed + ep if seed is not None else None
        s,_=env.reset(seed=ep_seed); done=False; steps=0
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

def evaluate_with_stats(agent, env_fn, episodes=20, seeds=20, seed=0):
    """Report mean +/- std over 20 eval seeds."""
    import numpy as np
    successes=[]
    steps=[]
    for s in range(seeds):
        env = env_fn(seed=seed+s)
        res = evaluate(agent, env, episodes=episodes, seed=seed+s)
        successes.append(res["success"])
        steps.append(res["avg_steps"])
    return {"mean_success": float(np.mean(successes)), "std_success": float(np.std(successes)), "mean_steps": float(np.mean(steps)), "std_steps": float(np.std(steps))}
