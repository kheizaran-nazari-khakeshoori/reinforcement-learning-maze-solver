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

def grid_to_walls(grid):
    walls=set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c]==1:
                walls.add((r,c))
    return walls

def get_learned_path(env, agent):
    path=[env.start]; s,_=env.reset()
    for _ in range(50):
        a=agent.act(s,0.0)
        ns,_,term,trunc,_=env.step(a)
        path.append(env.agent_pos)
        if term or trunc: break
        s=ns
    return path

def compare_paths(opt, learned):
    return {"opt":len(opt) if opt else 0, "learned":len(learned), "extra":len(learned)-len(opt) if opt else 0}

def manhattan(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
