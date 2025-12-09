from pyamaze import maze, agent
from collections import deque

def bfs_path(m, start=None, goal=None):
    if start is None:
        start = (m.rows, m.cols)  # pyamaze 默认入口
    if goal is None:
        goal = (1, 1)             # 默认出口

    maze_map = m.maze_map
    q = deque([start])
    parent = {start: None}

    dirs = {'N': (-1, 0),
            'S': (1, 0),
            'E': (0, 1),
            'W': (0, -1)}

    while q:
        cell = q.popleft()
        if cell == goal:
            break

        r, c = cell
        for d, (dr, dc) in dirs.items():
            if maze_map[(r, c)][d] == 1:          # 那个方向有路
                nr, nc = r + dr, c + dc
                if (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))

    # 回溯路径
    path = {}
    cur = goal
    while cur is not None:
        prev = parent[cur]
        if prev is not None:
            path[prev] = cur
        cur = prev
    return path   # 这是 pyamaze 需要的字典格式


# 使用
m = maze(15, 15)
m.CreateMaze()          # 生成迷宫

custom_path = bfs_path(m)   # 用自己的算法求路

a = agent(m, footprints=True, filled=True)
m.tracePath({a: custom_path})
m.run()
