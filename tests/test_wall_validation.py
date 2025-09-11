import pytest
from maze_solver.env import MazeEnv


def test_wall_overlap_start():
    with pytest.raises(ValueError):
        MazeEnv(size=3, walls=[(0, 0)])


def test_wall_overlap_goal():
    with pytest.raises(ValueError):
        MazeEnv(size=3, walls=[(2, 2)])


def test_valid_walls():
    env = MazeEnv(size=3, walls=[(1, 1)])
    assert (1, 1) in env.walls
