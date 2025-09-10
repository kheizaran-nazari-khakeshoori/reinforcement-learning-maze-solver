from maze_solver.env import MazeEnv


def test_is_terminal_goal():
    env = MazeEnv(size=3)
    assert env._is_terminal(env.goal)
    assert not env._is_terminal(env.start)


def test_is_terminal_trap():
    env = MazeEnv(size=3)
    env.traps = {(1, 1)}
    assert env._is_terminal((1, 1))
    assert not env._is_terminal((0, 1))
