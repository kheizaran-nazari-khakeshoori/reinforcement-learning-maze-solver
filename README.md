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

## Local Development
- Branch `local_commit` holds work-in-progress (10 small commits, not pushed yet)
- Helpers: `_pos_to_state`, `_state_to_pos`, `_is_terminal`, `DEFAULT_SIZE`
- Run `pytest tests/test_state_helpers.py tests/test_terminal.py tests/test_wall_validation.py`

## New local 30 commits
# Local 30b-1
## Changelog

- Local 30b batch 2
# End 30b
# 40-08
# 40-16
# 40-24
# 40-32
# 40-40
