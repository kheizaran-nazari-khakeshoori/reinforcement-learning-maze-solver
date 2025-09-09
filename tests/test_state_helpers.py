from maze_solver.env import MazeEnv


def test_state_pos_roundtrip():
    env = MazeEnv(size=4)
    for r in range(4):
        for c in range(4):
            s = env._pos_to_state((r, c))
            assert env._state_to_pos(s) == (r, c)


def test_state_bounds():
    env = MazeEnv(size=5)
    assert env._pos_to_state((0, 0)) == 0
    assert env._pos_to_state((4, 4)) == 24
    assert env._state_to_pos(0) == (0, 0)
