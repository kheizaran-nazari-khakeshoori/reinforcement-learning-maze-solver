# Reinforcement Learning Maze Solver

Gridworld agent learns to navigate, avoid traps, and reach the goal via the shortest path.

## Overview
- Custom Gymnasium grid environment (configurable size, walls, traps)
- Agents: Q-Learning (off-policy) and SARSA (on-policy)
- Training, evaluation, and visualization of shortest path

## Project Structure
```
maze_solver/  # env.py, agents.py
tests/        # pytest
pyproject.toml
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
pytest
```
