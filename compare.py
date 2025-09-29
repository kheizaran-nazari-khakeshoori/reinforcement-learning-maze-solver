"""Compare."""
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent, SarsaAgent
def run_compare(episodes=100):
    env=MazeEnv(size=4)
    q=QLearningAgent(env.observation_space.n, env.action_space.n)
    s=SarsaAgent(env.observation_space.n, env.action_space.n)
    return q,s
