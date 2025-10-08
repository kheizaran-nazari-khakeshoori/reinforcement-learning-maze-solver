from maze_solver.env import MazeEnv
def test_get_state():
    env=MazeEnv(size=3); env.reset(); assert env.get_state()==0
