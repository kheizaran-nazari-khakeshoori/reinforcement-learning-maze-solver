"""Path utilities: BFS optimal baseline and helpers."""

from collections import deque
from typing import List, Tuple, Set, Optional, Dict


def bfs(
    size: int,
    walls: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """Breadth-first search for shortest path."""
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nxt = (pos[0] + dr, pos[1] + dc)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in walls and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def grid_to_walls(grid: List[List[int]]) -> Set[Tuple[int, int]]:
    walls: Set[Tuple[int, int]] = set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                walls.add((r, c))
    return walls


def get_learned_path(env, agent) -> List[Tuple[int, int]]:
    """Greedy rollout from start using agent's policy."""
    path = [env.start]
    state, _ = env.reset()
    for _ in range(50):
        action = agent.act(state, 0.0)
        next_state, _, terminated, truncated, _ = env.step(action)
        path.append(env.agent_pos)
        if terminated or truncated:
            break
        state = next_state
    return path


def compare_paths(opt, learned) -> Dict[str, int]:
    return {
        "opt": len(opt) if opt else 0,
        "learned": len(learned),
        "extra": len(learned) - len(opt) if opt else 0,
    }


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_length(path) -> int:
    return len(path) - 1 if path else 0
