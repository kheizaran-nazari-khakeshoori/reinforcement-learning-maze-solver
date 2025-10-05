from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from evaluate import evaluate
def test_eval():
    env=MazeEnv(size=2)
    ag=QLearningAgent(4,4)
    ag.q_table[0,1]=10; ag.q_table[1,2]=10
    m=evaluate(ag,env, episodes=2)
    assert "success" in m
