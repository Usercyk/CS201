# Assignment #B: 图为主

Updated May 1, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：经典的普通的BFS

代码：

```python
# coding: utf-8
"""
@File        :   algernon_07218.py
@Time        :   2025/05/02 23:50:28
@Author      :   Usercyk
@Description :   BFS to find the shortest path in a grid with obstacles.
"""
from collections import deque
from typing import List, Optional, Tuple


class Solution:
    """
    The solution class
    """
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def solve_labyrinth(self,
                        r: int,
                        c: int,
                        start: Tuple[int, int],
                        end: Tuple[int, int],
                        labyrinth: List[List[str]]) -> Optional[int]:
        """
        Solve the labyrinth problem using BFS.

        Arguments:
            r -- row count of the labyrinth
            c -- column count of the labyrinth
            start -- the starting position (x, y) in the labyrinth
            end -- the ending position (x, y) in the labyrinth
            labyrinth -- the labyrinth grid
        """
        queue = deque([start])
        visited = [[False] * c for _ in range(r)]
        visited[start[0]][start[1]] = True
        distance = 0
        while queue:
            for _ in range(len(queue)):
                x, y = queue.popleft()
                if (x, y) == end:
                    return distance
                for dx, dy in self.DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < r and 0 <= ny < c:
                        if not visited[nx][ny] and labyrinth[nx][ny] != '#':
                            visited[nx][ny] = True
                            queue.append((nx, ny))
            distance += 1
        return None

    def input_labyrinth(self) -> Tuple[int, int, Tuple[int, int], Tuple[int, int], List[List[str]]]:
        """
        Input the labyrinth from standard input.

        Returns:
            r -- row count of the labyrinth
            c -- column count of the labyrinth
            start -- the starting position (x, y) in the labyrinth
            end -- the ending position (x, y) in the labyrinth
            labyrinth -- the labyrinth grid
        """
        r, c = map(int, input().split())
        labyrinth = []
        start, end = None, None
        for i in range(r):
            line = input().strip()
            if 'S' in line:
                start = (i, line.index('S'))
            if 'E' in line:
                end = (i, line.index('E'))
            labyrinth.append(list(line))
        if start is None or end is None:
            raise ValueError(
                "Start or end position not found in the labyrinth.")
        return r, c, start, end, labyrinth

    def solve(self) -> None:
        """
        Solve the labyrinth problem and print the result.
        """
        for _ in range(int(input())):
            result = self.solve_labyrinth(*self.input_labyrinth())
            if result is not None:
                print(result)
            else:
                print("oop!")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503000035.png)

### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：染色🤔

代码：

```python
# coding: utf-8
"""
@File        :   path_existence_query_3532.py
@Time        :   2025/05/03 00:04:25
@Author      :   Usercyk
@Description :   Path existence query
"""
from typing import List


class Solution:
    """
    The solution class
    """

    # pylint: disable=C0103
    def pathExistenceQueries(self,
                             n: int,
                             nums: List[int],
                             maxDiff: int,
                             queries: List[List[int]]) -> List[bool]:
        """
        Check if there exists a path between two indices in the array nums.

        Arguments:
            n -- number of elements in nums
            nums -- the array of integers
            maxDiff -- the maximum allowed difference between adjacent elements
            queries -- the list of queries, each query is a list of two integers [u, v]

        Returns:
            A list of boolean values indicating if a path exists for each query.
        """
        tag = [0]*n
        for i in range(1, n):
            tag[i] = tag[i-1] + int(abs(nums[i]-nums[i-1]) > maxDiff)
        return [tag[u] == tag[v] for u, v in queries]

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503001116.png)

### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：利用二分求b，每次利用二分找到85以上

代码：

```python
# coding: utf-8
"""
@File        :   adjust_score_22528.py
@Time        :   2025/05/03 00:15:08
@Author      :   Usercyk
@Description :   Adjust Score
"""
from bisect import bisect
from typing import List


class Solution:
    """
    The solution class
    """
    MAX_B = 1_000_000_000

    def is_ok(self, scores: List[float], n: int, b: int) -> bool:
        """
        Check if the score adjustment is valid
        """
        a = b/self.MAX_B
        ns = [a*x+1.1**(a*x) for x in scores]
        idx = bisect(ns, 85.0)
        return idx <= n*0.4

    def solve(self) -> None:
        """
        The main function to solve the problem
        """
        scores = sorted(map(float, input().split()))
        n = len(scores)
        l, r = 1, self.MAX_B
        while l < r:
            m = (l+r)//2
            if self.is_ok(scores, n, m):
                r = m
            else:
                l = m+1
        print(l)


if __name__ == '__main__':
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503002911.png)

### Msy382: 有向图判环

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：拓扑排序，排不了就说明有环

代码：

```python
# coding: utf-8
"""
@File        :   directed_graph_ring.py
@Time        :   2025/05/03 00:33:31
@Author      :   Usercyk
@Description :   Check if a directed graph contains a ring.
"""
from collections import deque


class Solution:
    """
    The solution class
    """

    def solve(self) -> None:
        """
        Solve the problem
        """
        n, m = map(int, input().split())
        graph = [[] for _ in range(n)]
        indegree = [0] * n
        for _ in range(m):
            u, v = map(int, input().split())
            graph[u].append(v)
            indegree[v] += 1

        queue = deque([i for i in range(n) if indegree[i] == 0])
        cnt = 0

        while queue:
            vtx = queue.popleft()
            cnt += 1
            for neighbor in graph[vtx]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if cnt == n:
            print("No")
        else:
            print("Yes")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503004041.png)

### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：Floyd-Warshall

代码：

```python
# coding: utf-8
"""
@File        :   rabbit_05443.py
@Time        :   2025/05/03 00:54:24
@Author      :   Usercyk
@Description :   Floyd-Warshall Algorithm for Rabbit Problem
"""


from itertools import product
from typing import List


class Solution:
    """
    The solution class
    """
    INF = 1 << 30

    def solve(self) -> None:
        """
        Solve the problem
        """
        n = int(input())
        locations = [input().strip() for _ in range(n)]

        distances = [[self.INF] * n
                     for _ in range(n)]
        for i in range(n):
            distances[i][i] = 0

        next_node = [[-1] * n for _ in range(n)]

        for _ in range(int(input())):
            a, b, d = input().split()
            u = locations.index(a)
            v = locations.index(b)
            distances[u][v] = min(distances[u][v], int(d))
            distances[v][u] = min(distances[v][u], int(d))
            next_node[u][v] = v
            next_node[v][u] = u

        for k, i, j in product(range(n), repeat=3):
            if distances[i][j] > distances[i][k] + distances[k][j]:
                distances[i][j] = distances[i][k] + distances[k][j]
                next_node[i][j] = next_node[i][k]

        for _ in range(int(input())):
            a, b = input().split()
            u = locations.index(a)
            v = locations.index(b)
            if distances[u][v] == self.INF:
                print("No path")
            else:
                path = self.reconstruct_path(
                    next_node, u, v, locations, distances)
                print(path)

    def reconstruct_path(self, next_node: List[List[int]],
                         u: int, v: int, locations: List[str], distances: List[List[int]]) -> str:
        """
        Reconstruct the path from u to v using the next_node matrix.

        Arguments:
            next_node -- the next node matrix
            u -- the starting node
            v -- the ending node
            locations -- the list of locations

        Returns:
            The path as a string
        """
        paths = []
        while u != v:
            paths.append(u)
            u = next_node[u][v]
        paths.append(v)

        path = locations[paths[0]]
        for i in range(1, len(paths)):
            path += f"->({distances[paths[i-1]][paths[i]]})->{locations[paths[i]]}"
        return path


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503011506.png)

### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：超时了……启发式搜索启动！

代码：

```python
# coding: utf-8
"""
@File        :   knights_tour_28050.py
@Time        :   2025/05/03 00:43:08
@Author      :   Usercyk
@Description :   Knights Tour Problem
"""

from typing import List, Tuple


class Solution:
    """
    The solution class
    """
    MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]

    def check(self, n: int, start: Tuple[int, int]) -> bool:
        """
        Check if the knight can visit all squares on the chessboard.

        Arguments:
            n -- size of the chessboard
            start -- starting position of the knight

        Returns:
            True if the knight can visit all squares, False otherwise
        """
        board = [[0] * n for _ in range(n)]
        x, y = start
        board[x][y] = 1

        return self.dfs(x, y, 1, board, n)

    def dfs(self, x: int, y: int, move_count: int, board: List[List[int]], n: int) -> bool:
        """
        Depth-first search to find a valid knight's tour.

        Arguments:
            x -- current x position of the knight
            y -- current y position of the knight
            move_count -- number of moves made so far
            board -- the chessboard
            n -- size of the chessboard

        Returns:
            True if a valid tour is found, False otherwise
        """
        if move_count == n * n:
            return True

        next_moves = []
        for dx, dy in self.MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == 0:
                count = 0
                for ddx, ddy in self.MOVES:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < n and 0 <= nny < n and board[nnx][nny] == 0:
                        count += 1
                next_moves.append((count, dx, dy))

        next_moves.sort()

        for count, dx, dy in next_moves:
            nx, ny = x + dx, y + dy
            board[nx][ny] = move_count + 1
            if self.dfs(nx, ny, move_count + 1, board, n):
                return True
            board[nx][ny] = 0

        return False

    def solve(self) -> None:
        """
        Solve the problem.
        """
        n = int(input())
        x, y = map(int, input().split())
        flag = self.check(n, (x, y))
        print("success" if flag else "fail")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250503005307.png)

## 2. 学习总结和收获

五一开卷！
