from maze_solver.agents import QLearningAgent
def test_repr():
    a=QLearningAgent(4,2); assert "QLearning" in repr(a)
