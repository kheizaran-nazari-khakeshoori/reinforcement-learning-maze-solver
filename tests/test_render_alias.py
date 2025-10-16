from maze_solver.env import MazeEnv
def test_alias():
    env=MazeEnv(size=2); assert "A" in env.render_text()
