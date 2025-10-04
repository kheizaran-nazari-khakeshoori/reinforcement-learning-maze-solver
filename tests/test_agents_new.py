from maze_solver.agents import QLearningAgent
def test_q():
    a=QLearningAgent(4,2)
    a.q_table[0,1]=1
    assert a.greedy_action(0)==1
