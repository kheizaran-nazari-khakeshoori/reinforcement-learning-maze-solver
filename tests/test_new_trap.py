from maze_solver.env import MazeEnv


def test_trap_penalty():
    env = MazeEnv(size=3, traps=[(0, 1)])
    env.reset()
    _, reward, term, _, _ = env.step(1)
    assert reward == -1.0
    assert term
    assert env._is_terminal((0, 1))
