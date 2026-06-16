"""Stronger baselines: A* and Dijkstra."""

import heapq
from typing import Set, Tuple, List, Optional

def astar(size: int, walls: Set[Tuple[int,int]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    open_set = [(0, start, [start])]
    g_score = {start: 0}
    visited=set()
    while open_set:
        f, pos, path = heapq.heappop(open_set)
        if pos == goal:
            return path
        if pos in visited: continue
        visited.add(pos)
        for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
            nxt=(pos[0]+dr,pos[1]+dc)
            if 0<=nxt[0]<size and 0<=nxt[1]<size and nxt not in walls:
                ng = g_score[pos]+1
                if nxt not in g_score or ng < g_score[nxt]:
                    g_score[nxt]=ng
                    h = abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1])
                    heapq.heappush(open_set,(ng+h, nxt, path+[nxt]))
    return None

def dijkstra(size: int, walls: Set[Tuple[int,int]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    pq=[(0,start,[start])]
    visited={}
    while pq:
        cost,pos,path=heapq.heappop(pq)
        if pos==goal:
            return path
        if pos in visited and visited[pos]<=cost: continue
        visited[pos]=cost
        for dr,dc in [(-1,0),(0,1),(1,0),(0,-1)]:
            nxt=(pos[0]+dr,pos[1]+dc)
            if 0<=nxt[0]<size and 0<=nxt[1]<size and nxt not in walls:
                heapq.heappush(pq,(cost+1,nxt,path+[nxt]))
    return None

# dijkstra baseline verified
