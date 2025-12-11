"""
迷宫寻宝实验（算法综合展示）
------------------------------------
支持以下算法在迷宫环境中规划路径并进行可视化展示：

1. 动态规划（DP）
2. 回溯法（Backtracking）
3. 贪心算法（改进版 Heuristic Greedy）
4. 分支限界（Branch and Bound）
5. BFS（作为最短路径参考，不参与评分）

主要实验目的：
- 比较不同算法在随机迷宫中的行为差异
- 分析不同策略在“吃金豆 + 走步数”得分系统下的表现
- 直观展示算法的搜索性质、优缺点

运行：
    pip install pygame
"""

import pygame
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Optional

# ----------------- 基本配置 -----------------
ROWS = 25
COLS = 25
CELL_SIZE = 24
PANEL_WIDTH = 260
TREASURE_COUNT = 8

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (245, 245, 245)
GRAY = (180, 180, 180)
DARK_GRAY = (40, 40, 40)
BLUE = (80, 160, 255)
GREEN = (80, 200, 120)
RED = (230, 80, 80)
YELLOW = (250, 210, 70)

# 格子类型
WALL = 0
ROAD = 1
START = 2
EXIT = 3
TREASURE = 4

Pos = Tuple[int, int]


# =========================================================
# 迷宫生成（DFS）
# =========================================================
def generate_empty_maze(rows: int, cols: int) -> List[List[int]]:
    return [[WALL for _ in range(cols)] for _ in range(rows)]


def generate_maze(rows: int, cols: int):
    """
    使用深度优先搜索（DFS）生成随机迷宫，
    并在通路上随机放置金豆（TREASURE）。
    """
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    grid = generate_empty_maze(rows, cols)

    # DFS 挖通路
    sr, sc = 1, 1
    grid[sr][sc] = ROAD
    stack = [(sr, sc)]
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        random.shuffle(dirs)
        carved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr][nc] == WALL:
                grid[r + dr // 2][c + dc // 2] = ROAD
                grid[nr][nc] = ROAD
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()

    # 起点终点
    start = (1, 1)
    ends = [(rows - 2, cols - 2), (1, cols - 2), (rows - 2, 1)]
    end = random.choice(ends)
    grid[start[0]][start[1]] = START
    grid[end[0]][end[1]] = EXIT

    # 随机放置金豆
    road_cells = [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if grid[r][c] == ROAD and (r, c) not in (start, end)
    ]
    random.shuffle(road_cells)
    treasures = road_cells[:TREASURE_COUNT]
    for r, c in treasures:
        grid[r][c] = TREASURE

    return grid, start, end, treasures


# =========================================================
# 基础函数：邻接 + 环路擦除
# =========================================================
def neighbors(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    ns = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            ns.append((nr, nc))
    return ns


def simplify_path(path):
    """
    【路径简化：环路擦除 Loop-Erasure】
    多种算法可能在探索过程中出现回头与环路。
    最终展示时需要不重复、无回头的“简单路径”。
    逻辑：
        - 当同一节点再次出现时，删除第一次出现后所有节点
    """
    pos_index = {}
    result = []

    for p in path:
        if p in pos_index:
            first = pos_index[p]
            # 删除中间节点
            to_remove = result[first+1:]
            for r in to_remove:
                pos_index.pop(r, None)
            result = result[:first+1]
        else:
            pos_index[p] = len(result)
            result.append(p)

    return result


# =========================================================
# BFS：只用于最短路径参考
# =========================================================
def bfs_shortest_path(grid, start, end):
    q = deque([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nb in neighbors(grid, *cur):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)

    if end not in parent:
        return []

    path = []
    cur = end
    while cur:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


# =========================================================
# 动态规划（DP）
# =========================================================
def dp_path(grid, start, end):
    """
    【动态规划思想】
    状态定义：
        (r, c, mask) ，mask 表示哪些金豆已被收集（bitmask）
    dp[state] = 当前状态下的最高得分

    得分模型（实验核心）：
        - 吃金豆： +20
        - 走一步： -1
    """

    rows, cols = len(grid), len(grid[0])

    # 给金豆编号
    treasure_index = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TREASURE:
                treasure_index[(r, c)] = idx
                idx += 1

    dp = {(start[0], start[1], 0): 0}
    parent = {(start[0], start[1], 0): None}

    changed = True
    while changed:
        changed = False
        for (r, c, mask), score in list(dp.items()):
            for nr, nc in neighbors(grid, r, c):
                new_mask = mask
                if (nr, nc) in treasure_index:
                    bit = 1 << treasure_index[(nr, nc)]
                    if not (mask & bit):
                        new_mask |= bit

                new_score = score - 1
                if new_mask != mask:
                    new_score += 20   # ★ 金豆奖励 20 分

                state = (nr, nc, new_mask)
                if state not in dp or new_score > dp[state]:
                    dp[state] = new_score
                    parent[state] = (r, c, mask)
                    changed = True

    # 找终点最高分
    best_state = None
    best_score = -10**9
    for (r, c, mask), s in dp.items():
        if (r, c) == end and s > best_score:
            best_state = (r, c, mask)
            best_score = s

    if best_state is None:
        return []

    # 回溯路径
    path = []
    cur = best_state
    while cur:
        r, c, mask = cur
        path.append((r, c))
        cur = parent[cur]

    return simplify_path(path[::-1])


# =========================================================
# 回溯法（Backtracking）
# =========================================================
def backtracking_path(grid, start, end):
    """
    【回溯思想】
    枚举所有可能路径，找到得分最高的方案。
    评分：
        吃金豆 +20
        走一步 -1

    特点：
        - 能找到绝对最优解
        - 但搜索空间巨大，仅适合小规模实验展示
    """

    rows, cols = len(grid), len(grid[0])

    treasure_index = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TREASURE:
                treasure_index[(r, c)] = idx
                idx += 1

    def bitcount(mask):
        return bin(mask).count("1")

    best_score = -10**9
    best_path = []
    visited = set()

    def dfs(r, c, mask, steps, path):
        nonlocal best_score, best_path

        # 到终点：计算得分
        if (r, c) == end:
            score = bitcount(mask) * 20 - steps
            if score > best_score:
                best_score = score
                best_path = path.copy()
            return

        visited.add((r, c, mask))

        for nr, nc in neighbors(grid, r, c):
            new_mask = mask
            if (nr, nc) in treasure_index:
                bit = 1 << treasure_index[(nr, nc)]
                if not (mask & bit):
                    new_mask |= bit

            state = (nr, nc, new_mask)
            if state in visited:
                continue

            dfs(nr, nc, new_mask, steps + 1, path + [(nr, nc)])

        visited.remove((r, c, mask))

    dfs(start[0], start[1], 0, 0, [start])
    return simplify_path(best_path)


# =========================================================
# 贪心算法（改进版）
# =========================================================
def greedy_search(grid, start, end):
    """
    【改进贪心算法 Heuristic Greedy】
    每一步根据评分函数选择最佳下一步：

        value = 金豆收益(20) - 移动代价(1)
                - BFS_距离终点 × 惩罚系数

    BFS 仅用于估计距离，不参与路径生成，使算法仍然是“局部最优”策略。

    为防止传统贪心走入死胡同，加入 fallback：
        若无可走方向 → 回退一步 → 继续搜索
    """

    rows, cols = len(grid), len(grid[0])

    treasures = {(r, c)
                 for r in range(rows)
                 for c in range(cols)
                 if grid[r][c] == TREASURE}

    # BFS 计算从某点到终点的最短距离（启发式）
    def bfs_dist(src):
        q = deque([src])
        dist = {src: 0}

        while q:
            cur = q.popleft()
            if cur == end:
                return dist[cur]
            for nb in neighbors(grid, *cur):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return 999999  # 无法到达终点

    bfs_cache = {}
    def get_dist(p):
        if p not in bfs_cache:
            bfs_cache[p] = bfs_dist(p)
        return bfs_cache[p]

    path = [start]
    visited = {start}
    cur = start

    while cur != end:

        candidates = []

        for nb in neighbors(grid, cur[0], cur[1]):
            if nb in visited:
                continue

            gain = 20 if nb in treasures else 0
            dist_end = get_dist(nb)

            value = gain - 1 - dist_end * 0.15
            candidates.append((value, nb))

        # 若无路 → fallback（回退一步）
        if not candidates:
            if len(path) >= 2:
                prev = path[-2]
                path.append(prev)
                cur = prev
                continue
            else:
                break

        candidates.sort(reverse=True)
        best_next = candidates[0][1]

        visited.add(best_next)
        path.append(best_next)
        cur = best_next

    return simplify_path(path)


# =========================================================
# 分支限界（Branch & Bound）
# =========================================================
def branch_and_bound(grid, start, end):
    import heapq

    rows, cols = len(grid), len(grid[0])

    treasure_index = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TREASURE:
                treasure_index[(r, c)] = idx
                idx += 1

    def bitcount(m):
        return bin(m).count("1")

    def evaluate(mask, steps):
        return bitcount(mask) * 20 - steps   # 分数

    def heuristic(p):
        return abs(p[0] - end[0]) + abs(p[1] - end[1])

    pq = []
    start_state = (start[0], start[1], 0, 0)
    heapq.heappush(pq, (0 + heuristic(start), start_state))

    best = {}
    parent = {start_state: None}
    goal = None

    while pq:
        f, (r, c, mask, steps) = heapq.heappop(pq)

        if (r, c) == end:
            goal = (r, c, mask, steps)
            break

        for nr, nc in neighbors(grid, r, c):

            new_mask = mask
            if (nr, nc) in treasure_index:
                new_mask |= 1 << treasure_index[(nr, nc)]

            new_steps = steps + 1
            new_score = evaluate(new_mask, new_steps)
            state = (nr, nc, new_mask, new_steps)

            if state not in best or new_score > best[state]:
                best[state] = new_score
                parent[state] = (r, c, mask, steps)
                heapq.heappush(pq, (-new_score + heuristic((nr, nc)), state))

    if goal is None:
        return []

    # 回溯
    path = []
    cur = goal
    while cur:
        r, c, mask, steps = cur
        path.append((r, c))
        cur = parent[cur]

    return simplify_path(path[::-1])


# =========================================================
# 算法入口
# =========================================================
def plan_path(grid, start, end, algo):
    if algo == "动态规划":
        return dp_path(grid, start, end)
    elif algo == "回溯法":
        return backtracking_path(grid, start, end)
    elif algo == "贪心算法":
        return greedy_search(grid, start, end)
    elif algo == "分支限界":
        return branch_and_bound(grid, start, end)
    else:
        return bfs_shortest_path(grid, start, end)


ALGORITHMS = ["动态规划", "回溯法", "贪心算法", "分支限界"]


# =========================================================
# 游戏主类：动画 + UI
# =========================================================
class MazeGame:
    def __init__(self):
        pygame.init()
        self.title_font = pygame.font.SysFont("Microsoft YaHei UI", 18, bold=True)
        self.font = pygame.font.SysFont("Microsoft YaHei UI", 18)
        self.small_font = pygame.font.SysFont("Microsoft YaHei UI", 14)

        self.maze_width = COLS * CELL_SIZE
        self.maze_height = ROWS * CELL_SIZE
        self.width = self.maze_width + PANEL_WIDTH
        self.height = self.maze_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("迷宫寻宝 - 算法演示系统")

        self.clock = pygame.time.Clock()

        # 游戏状态
        self.grid = []
        self.start = (1, 1)
        self.end = (ROWS - 2, COLS - 2)
        self.treasures = []
        self.initial_treasures = []
        self.player_pos = self.start

        self.current_algo_index = 0
        self.path = []
        self.path_index = 0
        self.animating = False

        # 统计
        self.shortest_len = 0
        self.cur_path_len = 0
        self.beans_total = 0
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0
        self.algorithm_time_ms = 0

        self.generate_new_maze()

    # ---------- 迷宫重置 ----------
    def generate_new_maze(self):
        self.grid, self.start, self.end, self.treasures = generate_maze(ROWS, COLS)
        self.initial_treasures = list(self.treasures)
        self.player_pos = self.start
        self.animating = False
        self.path = []
        self.path_index = 0

        self.beans_total = len(self.treasures)
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0

        walkable = [[ROAD if cell != WALL else WALL for cell in row] for row in self.grid]
        self.shortest_len = len(bfs_shortest_path(walkable, self.start, self.end))

    def reset_player(self):
        self.player_pos = self.start
        self.animating = False
        self.path_index = 0
        self.path = []
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0
        self.cur_path_len = 0
        self.algorithm_time_ms = 0

        # 恢复金豆
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == TREASURE:
                    self.grid[r][c] = ROAD
        for (r, c) in self.initial_treasures:
            self.grid[r][c] = TREASURE

    # ---------- 算法执行 ----------
    def start_demo(self):
        self.reset_player()
        algo_name = ALGORITHMS[self.current_algo_index]

        t0 = time.time()
        path = plan_path(self.grid, self.start, self.end, algo_name)
        t1 = time.time()

        self.path = path
        self.cur_path_len = len(path)
        self.algorithm_time_ms = round((t1 - t0) * 1000, 3)
        self.animating = True
        self.path_index = 0

    def update_animation(self, dt):
        if not self.animating or not self.path:
            return

        self.running_time += dt

        if self.path_index < len(self.path):
            self.player_pos = self.path[self.path_index]
            self.path_index += 1

            r, c = self.player_pos
            if self.grid[r][c] == TREASURE:
                self.grid[r][c] = ROAD
                self.beans_collected += 1

            if self.player_pos == self.end:
                self.animating = False
                self.compute_score()

    def compute_score(self):
        """
        最终得分：
            金豆 × 20  − 步数
        与所有算法的内部逻辑完全一致
        """
        steps = max(0, len(self.path) - 1)
        self.score = self.beans_collected * 20 - steps

    # ---------- 绘图 ----------
    def draw_maze(self):
        screen = self.screen

        for r in range(ROWS):
            for c in range(COLS):
                cell = self.grid[r][c]
                x, y = c * CELL_SIZE, r * CELL_SIZE

                if cell == WALL:
                    color = DARK_GRAY
                elif cell == ROAD:
                    color = WHITE
                elif cell == START:
                    color = (200, 240, 255)
                elif cell == EXIT:
                    color = (220, 255, 220)
                elif cell == TREASURE:
                    color = (255, 250, 210)
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

        # 网格线
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRAY, (0, r * CELL_SIZE), (self.maze_width, r * CELL_SIZE))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRAY, (c * CELL_SIZE, 0), (c * CELL_SIZE, self.maze_height))

        # 金豆
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == TREASURE:
                    cx = c * CELL_SIZE + CELL_SIZE // 2
                    cy = r * CELL_SIZE + CELL_SIZE // 2
                    pygame.draw.circle(screen, YELLOW, (cx, cy), CELL_SIZE // 4)

        # 玩家
        pr, pc = self.player_pos
        px = pc * CELL_SIZE + CELL_SIZE // 2
        py = pr * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, BLUE, (px, py), CELL_SIZE // 3)

        # 路径高亮
        if self.path_index > 1:
            pts = [
                (c * CELL_SIZE + CELL_SIZE // 2,
                 r * CELL_SIZE + CELL_SIZE // 2)
                for (r, c) in self.path[:self.path_index]
            ]
            if len(pts) > 1:
                pygame.draw.lines(screen, BLUE, False, pts, 3)

    def draw_panel(self):
        x0 = self.maze_width
        screen = self.screen

        pygame.draw.rect(screen, (235, 235, 235), (x0, 0, PANEL_WIDTH, self.height))

        card = pygame.Rect(x0 + 10, 20, PANEL_WIDTH - 20, self.height - 40)
        pygame.draw.rect(screen, (250, 250, 250), card, border_radius=16)
        pygame.draw.rect(screen, (210, 210, 210), card, 2, border_radius=16)

        y = card.y + 20

        title = self.title_font.render("控制面板", True, BLACK)
        screen.blit(title, (card.x + 20, y))
        y += 40

        algo = ALGORITHMS[self.current_algo_index]
        screen.blit(self.small_font.render("算法选择：", True, BLACK), (card.x + 20, y))
        screen.blit(self.small_font.render(f"[ {algo} ]", True, BLUE), (card.x + 100, y))
        y += 40

        ops = ["[G] 生成迷宫", "[S] 开始演示", "[R] 重置玩家", "[↑/↓] 切换算法", "[ESC] 退出"]
        for s in ops:
            screen.blit(self.small_font.render(s, True, BLACK), (card.x + 20, y))
            y += 22

        y += 20
        screen.blit(self.title_font.render("统计面板", True, BLACK), (card.x + 20, y))
        y += 40

        def stat(label, value):
            nonlocal y
            screen.blit(self.small_font.render(label, True, (90, 90, 90)), (card.x + 20, y))
            screen.blit(self.small_font.render(value, True, BLACK), (card.x + 120, y))
            y += 24

        stat("路径长度：", f"{self.cur_path_len} 步")
        stat("最短路径：", f"{self.shortest_len} 步")
        stat("算法时间：", f"{self.algorithm_time_ms} ms")
        stat("金豆收集：", f"{self.beans_collected} / {self.beans_total}")
        stat("当前得分：", f"{self.score}")

    # ---------- 主循环 ----------
    def run(self):
        running = True

        while running:
            dt = self.clock.tick(30) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        self.generate_new_maze()
                    elif event.key == pygame.K_r:
                        self.reset_player()
                    elif event.key == pygame.K_s:
                        self.start_demo()
                    elif event.key == pygame.K_UP:
                        self.current_algo_index = (self.current_algo_index - 1) % len(ALGORITHMS)
                    elif event.key == pygame.K_DOWN:
                        self.current_algo_index = (self.current_algo_index + 1) % len(ALGORITHMS)

            self.update_animation(dt)

            self.screen.fill(BLACK)
            self.draw_maze()
            self.draw_panel()
            pygame.display.flip()

        pygame.quit()


# =========================================================
# 程序入口
# =========================================================
if __name__ == "__main__":
    game = MazeGame()
    game.run()
