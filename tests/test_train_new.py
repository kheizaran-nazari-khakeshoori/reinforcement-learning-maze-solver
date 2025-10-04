from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from train import run_episode
def test_train():
    env=MazeEnv(size=3)
    ag=QLearningAgent(9,4)
    r,_=run_episode(env,ag,1.0)
    assert isinstance(r,float)
