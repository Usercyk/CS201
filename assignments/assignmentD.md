# Assignment #D: 图 & 散列表

Updated May 23, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```

思路：额……实现（

代码：

```python
# coding: utf-8
"""
@File        :   quadratic_probing_17975.py
@Time        :   2025/05/23 19:49:35
@Author      :   Usercyk
@Description :   Quadratic Probing
"""
import sys
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    def build_hash_table(self, m: int, num_list: List[int]) -> List[int]:
        """
        Build a hash table using quadratic probing

        Arguments:
            m -- the size of the hash table
            num_list -- the list of elements to be inserted

        Returns:
            A list representing the hash table
        """
        hash_table: List[Optional[int]] = [None] * m
        res = []
        for num in num_list:
            i = 0
            while True:
                index = (num + i * i) % m
                if hash_table[index] is None or hash_table[index] == num:
                    hash_table[index] = num
                    res.append(index)
                    break
                index = (num - i * i) % m
                if hash_table[index] is None or hash_table[index] == num:
                    hash_table[index] = num
                    res.append(index)
                    break
                i += 1
        return res

    def solve(self) -> None:
        """
        Solve the problem
        """
        data = sys.stdin.read().split()
        index = 0
        n = int(data[index])
        index += 1
        m = int(data[index])
        index += 1
        num_list = [int(i) for i in data[index:index+n]]

        print(*self.build_hash_table(m, num_list))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250523200829.png)

### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：不是，这个输入方式…………我不好评价

代码：

```python
# coding: utf-8
"""
@File        :   agri-net_01258.py
@Time        :   2025/05/23 20:10:42
@Author      :   Usercyk
@Description :   Build Agri-Net
"""
from itertools import islice
from typing import Iterable, List
import heapq


class Solution:
    """
    The solution class
    """

    def minimum_length(self, distances: List[List[int]], n: int) -> int:
        """
        Calculate the minimum length of the distances

        Arguments:
            distances -- the distances between the farms
            n -- the number of farms

        Returns:
            The minimum length of the MST
        """
        visited = [False] * n
        min_edge = [float('inf')] * n
        min_edge[0] = 0
        heap = [(0, 0)]  # (cost, node)
        total = 0

        while heap:
            cost, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            total += cost
            for v in range(n):
                if not visited[v] and distances[u][v] < min_edge[v]:
                    min_edge[v] = distances[u][v]
                    heapq.heappush(heap, (distances[u][v], v))
        return total

    def chunked_iter(self, iterable: Iterable, n: int) -> Iterable[List[int]]:
        """
        Chunk an iterable into chunks of size n

        Arguments:
            iterable -- the iterable to be chunked
            n -- the size of each chunk

        Returns:
            An iterable of chunks
        """
        it = iter(iterable)
        while True:
            chunk = list(islice(it, n))
            if not chunk:
                break
            yield chunk

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                n = int(input())
                lines = []
                while len(lines) < n * n:
                    line = map(int, input().split())
                    lines.extend(line)
                it = iter(lines)
                distances = list(self.chunked_iter(it, n))
                print(self.minimum_length(distances, n))
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250523205011.png)

### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：添加了传送门的bfs，为了保持顺序，左右append

代码：

```python
# coding: utf-8
"""
@File        :   grid_teleportation_traversal_3552.py
@Time        :   2025/05/23 21:01:20
@Author      :   Usercyk
@Description :   Grid Teleportation Traversal
"""

from collections import defaultdict, deque
from typing import List


class Solution:
    """
    The solution class
    """

    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    INF = float('inf')

    def min_moves(self, matrix: List[str]) -> int:
        """
        Calculate the minimum number of moves to traverse the grid

        Arguments:
            matrix -- the grid represented as a list of strings

        Returns:
            The minimum number of moves to traverse the grid
        """
        m, n = len(matrix), len(matrix[0])
        if matrix[m-1][n-1] == '#':
            return -1

        tele = defaultdict(list)
        for i in range(m):
            for j in range(n):
                if matrix[i][j].isupper():
                    tele[matrix[i][j]].append((i, j))

        distance = [[self.INF] * n for _ in range(m)]
        distance[0][0] = 0
        q = deque([(0, 0)])

        while q:
            x, y = q.popleft()
            dist = distance[x][y]
            if x == m-1 and y == n-1:
                return int(dist)

            for dx, dy in self.DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#':
                    if dist+1 < distance[nx][ny]:
                        distance[nx][ny] = dist + 1
                        q.append((nx, ny))

            if matrix[x][y] in tele:
                for tx, ty in tele.pop(matrix[x][y]):
                    if dist <= distance[tx][ty]:
                        distance[tx][ty] = dist
                        q.appendleft((tx, ty))

        return -1

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250523211706.png)

### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：只运行k+1遍的bellman

代码：

```python
# coding: utf-8
"""
@File        :   cheapest_flight_within_k_stops_787.py
@Time        :   2025/05/24 14:26:21
@Author      :   Usercyk
@Description :   Find the cheapest price for a flight from src to dst with at most k stops.
"""
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    INF = float('inf')

    def find_cheapest_price(self,
                            n: int,
                            flights: List[List[int]],
                            src: int, dst: int, k: int) -> int:
        """
        Find the cheapest price for a flight from src to dst with at most k stops.

        Arguments:
            n -- number of cities
            flights -- list of flights
                flights[i] = [from, to, price]
                from -- the starting city
                to -- the destination city
                price -- the price of the flight
            src -- the starting city
            dst -- the destination city
            k -- the maximum number of stops

        Returns:
            the cheapest price for a flight from src to dst with at most k stops
                if there is no such flight, return -1
        """
        prices = [self.INF] * n
        prices[src] = 0
        predecessors: List[Optional[int]] = [None] * n

        for _ in range(k+1):
            updated = False
            p = prices.copy()
            for u, v, w in flights:
                if prices[u] != self.INF and p[u] + w < prices[v]:
                    prices[v] = p[u] + w
                    predecessors[v] = u
                    updated = True
            if not updated:
                break

        return int(prices[dst]) if prices[dst] != self.INF else -1

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250524143807.png)

### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：看了半天，题目没说c的范围……我不太清楚c会不会有负数，有负数的话就不能直接用dijkstra了。

但是只用SPFA又超时好几回，最后加了一个SLF优化才过。卡SPFA的常数可还行

代码：

```python
# coding: utf-8
"""
@File        :   candies_03424.py
@Time        :   2025/05/24 14:59:29
@Author      :   Usercyk
@Description :   Candies
"""
from collections import deque


class Solution:
    """
    The solution class
    """

    INF = float("inf")

    def solve(self) -> None:
        """
        Solve the problem
        """
        n, m = map(int, input().split())

        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            a, b, c = map(int, input().split())
            adj[a].append((b, c))

        distances = [self.INF] * (n+1)
        distances[1] = 0

        visited = [False] * (n+1)
        visited[1] = True
        q = deque([1])

        while q:
            u = q.popleft()
            visited[u] = False
            for (v, c) in adj[u]:
                if distances[u] + c < distances[v]:
                    distances[v] = distances[u] + c
                    if not visited[v]:
                        visited[v] = True
                        if q and distances[v] < distances[q[0]]:
                            q.appendleft(v)
                        else:
                            q.append(v)

        print(distances[n])


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250524151539.png)

### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：拓扑排序，代码基本上没改

代码：

```python
# coding: utf-8
"""
@File        :   minimum_award_22508.py
@Time        :   2025/05/24 14:42:18
@Author      :   Usercyk
@Description :   Calculate the minimum award
"""
from typing import List, Tuple


class Vertex:
    """
    The vertex
    """

    def __init__(self, tag: int) -> None:
        self.tag: int = tag
        self.in_degree: int = 0
        self.out_degree: int = 0
        self.out_adjacent: List["Vertex"] = []

    def connect_to(self, other: "Vertex"):
        """
        Connect this vertex to another vertex

        Arguments:
            other -- The other vertex

        Returns:
            The other vertex
        """
        self.out_adjacent.append(other)
        self.out_degree += 1
        other.in_degree += 1


class Graph:
    """
    The graph
    """

    def __init__(self, n: int) -> None:
        self._vertices: List[Vertex] = [Vertex(i+1) for i in range(n)]

    def connect(self, u: int, v: int) -> None:
        """
        Connect u to v

        Arguments:
            u -- The first vertex
            v -- The second vertex
        """
        self._vertices[u].connect_to(self._vertices[v])

    def minimum_award(self) -> int:
        """
        Calculate the minimum award

        Returns:
            The minimum award
        """
        queue: List[Tuple[Vertex, int]] = [
            (v, 100) for v in self._vertices if v.in_degree == 0]
        award = 0
        while queue:
            vertex, i = queue.pop(0)
            award += i
            for adj in vertex.out_adjacent:
                adj.in_degree -= 1
                if adj.in_degree == 0:
                    queue.append((adj, i+1))
        return award


class Solution:
    """
    The solution class
    """

    def solve(self) -> None:
        """
        Solve the problem
        """
        v, a = map(int, input().split())
        graph = Graph(v)
        for _ in range(a):
            v, u = map(int, input().split())
            graph.connect(u, v)
        print(graph.minimum_award())


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250524144811.png)

## 2. 学习总结和收获

还没做几个每日选做就要期末考试了，/(ㄒoㄒ)/~~
