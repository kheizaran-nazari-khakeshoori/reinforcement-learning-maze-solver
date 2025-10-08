from maze_solver.agents import QLearningAgent
def test_reset():
    a=QLearningAgent(4,2); a.q_table[0,0]=5; a.reset(); assert (a.q_table==0).all()
