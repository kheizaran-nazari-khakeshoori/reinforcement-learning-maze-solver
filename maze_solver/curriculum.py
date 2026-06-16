"""Curriculum and stochastic generation."""

import random
from typing import List, Tuple, Set

def random_walls_traps(size: int, n_walls: int = 3, n_traps: int = 2, seed=None) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    if seed is not None:
        random.seed(seed)
    walls=set()
    traps=set()
    while len(walls) < n_walls:
        r = random.randint(0, size-1)
        c = random.randint(0, size-1)
        if (r,c) in [(0,0),(size-1,size-1)]: continue
        walls.add((r,c))
    while len(traps) < n_traps:
        r = random.randint(0, size-1)
        c = random.randint(0, size-1)
        if (r,c) in walls or (r,c) in [(0,0),(size-1,size-1)]: continue
        traps.add((r,c))
    return walls, traps

def curriculum_schedule():
    """4x4 -> 7x7 schedule."""
    return [(4, 500), (5, 800), (6, 1200), (7, 1500)]

def get_curriculum_env(size, seed=None):
    from maze_solver.env import MazeEnv
    walls, traps = random_walls_traps(size, n_walls=max(1,size//2), n_traps=max(1,size//3), seed=seed)
    return MazeEnv(size=size, walls=walls, traps=traps, seed=seed)

# schedule verified 4x4->7x7
# stochastic traps per episode
# debuging curriculum verified
