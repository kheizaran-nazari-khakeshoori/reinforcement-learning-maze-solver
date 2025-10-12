from maze_solver.env import MazeEnv
def test_dist():
    env=MazeEnv(size=5); assert env.distance_to_goal()==8
