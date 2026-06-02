import subprocess
import time
import numpy as np
import pygame
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from train import train

 # Train agent with seed 0 for reproducibility
env = MazeEnv(size=5)
agent = QLearningAgent(env.observation_space.n, env.action_space.n)
print("Training 2000 episodes...")
train(agent, env, episodes=2000)
print("Training done, recording...")

s, _ = env.reset()
env.render(mode="pygame")  # creates env.gui.screen (maze_solver/env.py:100, maze_solver/gui.py:12)
w, h = env.gui.screen.get_size()
print(f"Screen {w}x{h}, recording to maze.mp4")

# Fedora ffmpeg has no libx264, use mpeg4 (tested: works with your ffmpeg 8.1.2)
cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{w}x{h}", "-r", "10", "-i", "-",
    "-c:v", "mpeg4", "-qscale:v", "5",
    "-pix_fmt", "yuv420p", "maze.mp4"
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

while True:
    if not env.render(mode="pygame"):
        break
    # capture frame from pygame surface
    arr = pygame.surfarray.array3d(env.gui.screen)  # (w, h, 3)
    arr = np.transpose(arr, (1, 0, 2))  # (h, w, 3)
    proc.stdin.write(arr.tobytes())

    a = agent.act(s, 0.0)
    s, _, term, trunc, _ = env.step(a)
    if term or trunc:
        # hold last frame a bit
        for _ in range(5):
            env.render(mode="pygame")
            arr = pygame.surfarray.array3d(env.gui.screen)
            proc.stdin.write(np.transpose(arr, (1, 0, 2)).tobytes())
            time.sleep(0.1)
        break
    time.sleep(0.2)

proc.stdin.close()
proc.wait()
print(f"Saved maze.mp4 (exit {proc.returncode})")
# Play with: ffplay maze.mp4  or  vlc maze.mp4
# Convert to h264 if needed: ffmpeg -i maze.mp4 -c:v libopenh264 maze_h264.mp4
