from maze_solver.env import MazeEnv


def test_movement_basic():
    env = MazeEnv(size=3)
    env.reset()
    env.step(1)
    assert env.agent_pos == (0, 1)


def test_boundary():
    env = MazeEnv(size=3)
    env.reset()
    env.step(3)
    assert env.agent_pos == (0, 0)
