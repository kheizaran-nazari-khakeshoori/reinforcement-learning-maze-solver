"""Utils."""
from collections import deque
def bfs(size, walls, start, goal):
    q=deque([(start,[start])]); vis={start}
    while q:
        pos,path=q.popleft()
        if pos==goal: return path
        for dr,dc in [(-1,0),(0,1),(1,0),(0,-1)]:
            nxt=(pos[0]+dr,pos[1]+dc)
            if 0<=nxt[0]<size and 0<=nxt[1]<size and nxt not in walls and nxt not in vis:
                vis.add(nxt); q.append((nxt, path+[nxt]))
    return None
