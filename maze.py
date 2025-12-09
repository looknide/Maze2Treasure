"""
maze_game.py
PyGame 迷宫寻宝模板：
- 左侧迷宫绘制区
- 右侧控制和统计面板
- 迷宫生成（DFS）
- 金豆随机放置
- BFS 最短路径规划 + 动画演示
- 简单积分系统

运行前先安装：
    pip install pygame
"""

import pygame
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Optional

# ----------------- 基本配置 -----------------
ROWS = 25            # 迷宫行数（奇数）
COLS = 25            # 迷宫列数（奇数）
CELL_SIZE = 24       # 每格像素
PANEL_WIDTH = 260    # 右侧控制面板宽度
TREASURE_COUNT = 8   # 金豆个数

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


# ----------------- 迷宫生成：DFS 回溯 -----------------
def generate_empty_maze(rows: int, cols: int) -> List[List[int]]:
    grid = [[WALL for _ in range(cols)] for _ in range(rows)]
    return grid


def generate_maze(rows: int, cols: int) -> Tuple[List[List[int]], Pos, Pos, List[Pos]]:
    """生成迷宫 + 起点、终点、金豆位置"""
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    grid = generate_empty_maze(rows, cols)

    # 1. DFS 挖迷宫
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

    # 2. 放置起点和终点（四角可选）
    start = (1, 1)
    end_candidates = [(rows - 2, cols - 2), (1, cols - 2),
                      (rows - 2, 1)]
    end = random.choice(end_candidates)
    grid[start[0]][start[1]] = START
    grid[end[0]][end[1]] = EXIT

    # 3. 放置金豆在通路上
    road_cells = [(r, c) for r in range(1, rows - 1)
                  for c in range(1, cols - 1)
                  if grid[r][c] == ROAD and (r, c) not in (start, end)]
    random.shuffle(road_cells)
    treasures: List[Pos] = road_cells[:TREASURE_COUNT]
    for tr, tc in treasures:
        grid[tr][tc] = TREASURE

    return grid, start, end, treasures


# ----------------- 路径规划：BFS 最短路径 -----------------
def neighbors(grid: List[List[int]], r: int, c: int) -> List[Pos]:
    rows, cols = len(grid), len(grid[0])
    res = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            res.append((nr, nc))
    return res


def bfs_shortest_path(grid: List[List[int]], start: Pos, end: Pos) -> List[Pos]:
    """返回从 start 到 end 的最短路径（包含起点和终点），不可达则空"""
    q = deque([start])
    parent: Dict[Pos, Optional[Pos]] = {start: None}

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
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# 预留：其它算法接口（目前都调用 BFS，可按实验要求自行替换）
def plan_path(grid: List[List[int]], start: Pos, end: Pos, algo: str) -> List[Pos]:
    if algo == "BFS(最短路)":
        return bfs_shortest_path(grid, start, end)
    # 下面这些先用 BFS 占位，你可以替换成真正的实现
    elif algo == "动态规划":
        return bfs_shortest_path(grid, start, end)
    elif algo == "回溯法":
        return bfs_shortest_path(grid, start, end)
    elif algo == "贪心算法":
        return bfs_shortest_path(grid, start, end)
    elif algo == "A*算法":
        return bfs_shortest_path(grid, start, end)
    elif algo == "分支限界":
        return bfs_shortest_path(grid, start, end)
    else:
        return bfs_shortest_path(grid, start, end)


ALGORITHMS = [
    "BFS(最短路)",
    "动态规划",
    "回溯法",
    "贪心算法",
    "A*算法",
    "分支限界",
]


# ----------------- 游戏主类 -----------------
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
        pygame.display.set_caption("迷宫寻宝")

        self.clock = pygame.time.Clock()

        # 游戏状态
        self.grid: List[List[int]] = []
        self.start: Pos = (1, 1)
        self.end: Pos = (ROWS - 2, COLS - 2)
        self.treasures: List[Pos] = []
        self.player_pos: Pos = self.start

        self.current_algo_index = 0
        self.path: List[Pos] = []
        self.path_index = 0
        self.animating = False

        # 统计与积分
        self.shortest_len = 0
        self.cur_path_len = 0
        self.beans_total = 0
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0
        self.start_time = 0.0

        self.generate_new_maze()

    # ---------- 迷宫与游戏状态 ----------
    def generate_new_maze(self):
        self.grid, self.start, self.end, self.treasures = generate_maze(ROWS, COLS)
        self.player_pos = self.start
        self.animating = False
        self.path = []
        self.path_index = 0

        self.beans_total = len(self.treasures)
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0

        # 预先计算真正最短路径长度（基于 BFS）
        true_grid = [[ROAD if cell != WALL else WALL for cell in row] for row in self.grid]
        self.shortest_len = len(bfs_shortest_path(true_grid, self.start, self.end))

    def reset_player(self):
        self.player_pos = self.start
        self.animating = False
        self.path_index = 0
        self.path = []
        self.beans_collected = 0
        self.score = 0
        self.running_time = 0.0

        # 重置金豆
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == TREASURE:
                    # 保持为宝藏
                    pass

    # ---------- 算法与动画 ----------
    def start_demo(self):
        """根据当前算法计算路径并开始动画"""
        algo_name = ALGORITHMS[self.current_algo_index]

        # 用一个 copy 的 grid，把 START/EXIT 视为 ROAD
        simple_grid = [[ROAD if cell != WALL else WALL for cell in row] for row in self.grid]

        t0 = time.time()
        path = plan_path(simple_grid, self.start, self.end, algo_name)
        t1 = time.time()

        self.path = path
        self.cur_path_len = len(path)
        self.start_time = time.time()
        self.running_time = 0.0
        self.animating = True
        self.path_index = 0

        # 执行时间只是算法计算路径的时间
        self.algorithm_time_ms = int((t1 - t0) * 1000)

    def update_animation(self, dt: float):
        if not self.animating or not self.path:
            return

        self.running_time += dt

        # 每帧走一步（可以改成基于时间的速度）
        if self.path_index < len(self.path):
            self.player_pos = self.path[self.path_index]
            self.path_index += 1

            # 收集金豆
            r, c = self.player_pos
            if self.grid[r][c] == TREASURE:
                self.grid[r][c] = ROAD
                self.beans_collected += 1

            # 到达终点
            if self.player_pos == self.end:
                self.animating = False
                self.compute_score()

    def compute_score(self):
        """根据积分规则计算得分"""
        beans_score = self.beans_collected * 10

        all_beans_bonus = 50 if self.beans_collected == self.beans_total else 0

        # 时间奖励：越快越高，最长 30 分
        time_bonus = max(0, 30 - int(self.running_time))

        optimal_bonus = 0
        over_penalty = 0
        if self.shortest_len > 0 and self.cur_path_len > 0:
            if self.cur_path_len == self.shortest_len:
                optimal_bonus = 30
            elif self.cur_path_len > self.shortest_len:
                over_penalty = self.cur_path_len - self.shortest_len

        self.score = beans_score + all_beans_bonus + time_bonus + optimal_bonus - over_penalty

    # ---------- 绘制 ----------
    def draw_maze(self):
        for r in range(ROWS):
            for c in range(COLS):
                cell = self.grid[r][c]
                x = c * CELL_SIZE
                y = r * CELL_SIZE

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
                else:
                    color = WHITE

                pygame.draw.rect(self.screen, color,
                                 (x, y, CELL_SIZE, CELL_SIZE))

        # 网格线
        for r in range(ROWS + 1):
            y = r * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (0, y), (self.maze_width, y), 1)
        for c in range(COLS + 1):
            x = c * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.maze_height), 1)

        # 金豆图标
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == TREASURE:
                    cx = c * CELL_SIZE + CELL_SIZE // 2
                    cy = r * CELL_SIZE + CELL_SIZE // 2
                    pygame.draw.circle(self.screen, YELLOW, (cx, cy), CELL_SIZE // 4)

        # 玩家
        pr, pc = self.player_pos
        px = pc * CELL_SIZE + CELL_SIZE // 2
        py = pr * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, BLUE, (px, py), CELL_SIZE // 3)

        # 起点终点标记
        sr, sc = self.start
        er, ec = self.end
        sx = sc * CELL_SIZE + CELL_SIZE // 2
        sy = sr * CELL_SIZE + CELL_SIZE // 2
        ex = ec * CELL_SIZE + CELL_SIZE // 2
        ey = er * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, GREEN, (sx, sy), CELL_SIZE // 5)
        pygame.draw.circle(self.screen, RED, (ex, ey), CELL_SIZE // 5)

        # 路径高亮（用浅蓝色实线）
        if self.path:
            points = []
            for (r, c) in self.path[:self.path_index]:
                x = c * CELL_SIZE + CELL_SIZE // 2
                y = r * CELL_SIZE + CELL_SIZE // 2
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, BLUE, False, points, 3)

    def draw_panel(self):
        x0 = self.maze_width

        # 面板整体背景
        pygame.draw.rect(self.screen, (235, 235, 235),
                         (x0, 0, PANEL_WIDTH, self.height))

        # 内层圆角卡片
        card_margin_x = 10
        card_margin_y = 20
        card_width = PANEL_WIDTH - card_margin_x * 2
        card_height = self.height - card_margin_y * 2
        card_rect = pygame.Rect(x0 + card_margin_x,
                                card_margin_y,
                                card_width,
                                card_height)
        pygame.draw.rect(self.screen, (250, 250, 250), card_rect, border_radius=16)
        pygame.draw.rect(self.screen, (210, 210, 210), card_rect, width=2, border_radius=16)

        # 为了让两个板块看起来垂直居中，按预估高度计算起始 y
        control_block_h = 160
        stats_block_h = 160
        gap_h = 40
        total_h = control_block_h + stats_block_h + gap_h
        base_y = card_rect.y + (card_rect.height - total_h) // 2

        # ---------- 控制面板 ----------
        y = base_y

        # 标题左侧小方块图标
        icon_size = 14
        icon_rect = pygame.Rect(card_rect.x + 18, y + 6, icon_size, icon_size)
        pygame.draw.rect(self.screen, (80, 150, 255), icon_rect, border_radius=3)

        title = self.title_font.render("控制面板", True, BLACK)
        self.screen.blit(title, (card_rect.x + 18 + icon_size + 8, y))

        y += 40

        # 算法选择
        algo_label = self.small_font.render("算法选择：", True, BLACK)
        self.screen.blit(algo_label, (card_rect.x + 20, y))

        algo_name = ALGORITHMS[self.current_algo_index]
        algo_text = self.small_font.render(f"[ {algo_name} ]", True, BLUE)
        self.screen.blit(algo_text, (card_rect.x + 90, y))

        y += 20

        # 按钮提示（用纯文本）
        btn1 = self.small_font.render("[G] 生成迷宫", True, BLACK)
        btn2 = self.small_font.render("[S] 开始演示", True, BLACK)
        btn3 = self.small_font.render("[R] 重置玩家", True, BLACK)
        btn4 = self.small_font.render("[ESC] 退出游戏", True, BLACK)
        btn5 = self.small_font.render("[↑/↓] 切换算法", True, BLACK)
        self.screen.blit(btn1, (card_rect.x + 20, y))
        self.screen.blit(btn2, (card_rect.x + 20, y + 20))
        self.screen.blit(btn3, (card_rect.x + 20, y + 40))
        self.screen.blit(btn4, (card_rect.x + 20, y + 60))
        self.screen.blit(btn5, (card_rect.x + 20, y + 80))

        # ---------- 统计面板 ----------
        y = base_y + control_block_h + gap_h

        stats_icon_rect = pygame.Rect(card_rect.x + 18, y + 6, icon_size, icon_size)
        pygame.draw.rect(self.screen, (100, 200, 120), stats_icon_rect, border_radius=3)

        stats_title = self.title_font.render("统计面板", True, BLACK)
        self.screen.blit(stats_title, (card_rect.x + 18 + icon_size + 8, y))

        y += 40
        line_y = y

        def stat_line(label: str, value: str):
            nonlocal line_y
            label_txt = self.small_font.render(label, True, (90, 90, 90))
            value_txt = self.small_font.render(value, True, BLACK)
            self.screen.blit(label_txt, (card_rect.x + 20, line_y))
            self.screen.blit(value_txt, (card_rect.x + 120, line_y))
            line_y += 24

        stat_line("路径长度：", f"{self.cur_path_len} 步")
        stat_line("最短路径：", f"{self.shortest_len} 步")
        stat_line("算法时间：", f"{getattr(self, 'algorithm_time_ms', 0)} ms")
        # stat_line("通关时间：", f"{self.running_time:.1f} 秒")
        stat_line("金豆收集：", f"{self.beans_collected} / {self.beans_total}")
        stat_line("当前得分：", f"{self.score}")

    # ---------- 主循环 ----------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(30) / 1000.0  # 秒
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        self.generate_new_maze()
                    elif event.key == pygame.K_s:
                        self.start_demo()
                    elif event.key == pygame.K_r:
                        self.reset_player()
                    elif event.key == pygame.K_UP:
                        self.current_algo_index = (self.current_algo_index - 1) % len(ALGORITHMS)
                    elif event.key == pygame.K_DOWN:
                        self.current_algo_index = (self.current_algo_index + 1) % len(ALGORITHMS)

            # 更新动画
            self.update_animation(dt)

            # 绘制
            self.screen.fill(BLACK)
            self.draw_maze()
            self.draw_panel()

            pygame.display.flip()

        pygame.quit()


# ----------------- 入口 -----------------
if __name__ == "__main__":
    game = MazeGame()
    game.run()
