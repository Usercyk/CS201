# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by 曹以楷 物理学院

AC6

## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：排序

代码：

```python
# coding: utf-8
"""
@File        :   election_06364.py
@Time        :   2025/05/16 19:19:42
@Author      :   Usercyk
@Description :   Cow Election
"""
from typing import List, Tuple


class Solution:
    """
    The solution class
    """

    def elect(self) -> None:
        """
        Choose first k candidates, then choose the one with the most votes
        """
        n, k = map(int, input().split())
        a: List[Tuple[int, int]] = [
            tuple(map(int, input().split())) for _ in range(n)]  # type: ignore
        b: List[Tuple[int, Tuple[int, int]]] = sorted(
            enumerate(a), key=lambda x: x[1], reverse=True)
        b = b[:k]
        b.sort(key=lambda x: x[1][1], reverse=True)
        print(b[0][0] + 1)


if __name__ == "__main__":
    Solution().elect()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516192428.png)

### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：按提示即可

代码：

```python
# coding: utf-8
"""
@File        :   all_possible_stack_sequence_04077.py
@Time        :   2025/05/16 19:27:54
@Author      :   Usercyk
@Description :   All Possible Stack Sequence
"""


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.n = 0
        self.ans = 0

    def dfs(self, a: int, b: int, step: int) -> None:
        """
        The dfs function

        Arguments:
            a -- Outside number
            b -- Inside number
            step -- Step number
        """
        if step == self.n:
            self.ans += 1
            return
        if a > 0:
            self.dfs(a-1, b+1, step)
        if b > 0:
            self.dfs(a, b-1, step+1)

    def all_possible_stack_sequence(self) -> None:
        """
        The main function
        """
        self.n = int(input())
        self.ans = 0
        self.dfs(self.n, 0, 0)
        print(self.ans)


if __name__ == "__main__":
    Solution().all_possible_stack_sequence()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516193206.png)

### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：实现………

代码：

```python
# coding: utf-8
"""
@File        :   card_sort_05343.py
@Time        :   2025/05/16 19:33:31
@Author      :   Usercyk
@Description :   Sort the cards
"""
from collections import defaultdict


class Solution:
    """
    The solution class
    """

    def sort_cards(self) -> None:
        """
        The main function
        """
        input()
        cards = input().split()
        d1 = defaultdict(list)
        cards1 = []
        for card in cards:
            d1[card[1]].append(card)
        for k in range(1, 10):
            print(f"Queue{k}:"+" ".join(d1[str(k)]))
            cards1.extend(d1[str(k)])
        d2 = defaultdict(list)
        cards2 = []
        for card in cards1:
            d2[card[0]].append(card)
        for k in ("A", "B", "C", "D"):
            print(f"Queue{k}:"+" ".join(d2[k]))
            cards2.extend(d2[k])
        print(*cards2)


if __name__ == "__main__":
    Solution().sort_cards()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516194101.png)

### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：把入度为1的点拿出去，然后把新的入度为1点拿出去……

代码：

```python
# coding: utf-8
"""
@File        :   topological_sort_04084.py
@Time        :   2025/05/16 19:42:12
@Author      :   Usercyk
@Description :   Topological Sort
"""
from typing import List


class Vertex:
    """
    The vertex
    """

    def __init__(self, tag: int) -> None:
        self.tag: int = tag
        self.in_degree: int = 0
        self.out_degree: int = 0
        self.out_adjacent: List["Vertex"] = []

    def __str__(self) -> str:
        return f"v{self.tag}"

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

    def __lt__(self, other: "Vertex") -> bool:
        """
        Compare two vertices

        Arguments:
            other -- The other vertex

        Returns:
            True if this vertex is less than the other vertex
        """
        return self.tag < other.tag


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

    def topological_sort(self) -> List[str]:
        """
        Topological sort the graph

        Returns:
            The topological order of the graph
        """
        order: List[str] = []
        queue: List[Vertex] = [v for v in self._vertices if v.in_degree == 0]
        while queue:
            vertex = queue.pop(0)
            order.append(str(vertex))
            for adj in vertex.out_adjacent:
                adj.in_degree -= 1
                if adj.in_degree == 0:
                    queue.append(adj)
            queue.sort()
        return order


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
            u, v = map(int, input().split())
            graph.connect(u-1, v-1)
        order = graph.topological_sort()
        print(*order)


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516200051.png)


### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：在普通dijkstra的基础上多存一个费用信息

代码：

```python
# coding: utf-8
"""
@File        :   road_07735.py
@Time        :   2025/05/16 20:12:04
@Author      :   Usercyk
@Description :   Dijkstra to find the shortest path
"""
from heapq import heappop, heappush


class Solution:
    """
    The solution class
    """
    INF = float("inf")

    def solve(self) -> None:
        """
        Solve the problem
        """
        k = int(input())
        n = int(input())
        r = int(input())

        adjacency_matrix = [[] for _ in range(n + 1)]
        for _ in range(r):
            s, d, l, t = map(int, input().split())
            adjacency_matrix[s].append((d, l, t))

        distances = [[self.INF] * (k + 1) for _ in range(n + 1)]
        distances[1][0] = 0

        heap = [(0, 1, 0)]
        flag = False

        while heap:
            current_distance, current_city, current_cost = heappop(heap)
            if current_city == n:
                print(current_distance)
                flag = True
                break
            if current_distance > distances[current_city][current_cost]:
                continue
            for next_city, length, cost in adjacency_matrix[current_city]:
                new_cost = current_cost + cost
                if new_cost > k:
                    continue
                new_distance = current_distance + length
                if new_distance < distances[next_city][new_cost]:
                    distances[next_city][new_cost] = new_distance
                    heappush(heap, (new_distance, next_city, new_cost))

        if not flag:
            print("-1")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516202725.png)

### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：爷爷+4个孙子与两个儿子的PK，什么隔代亲（不是

代码：

```python
# coding: utf-8
"""
@File        :   treasure_binary_tree_24637.py
@Time        :   2025/05/16 20:32:27
@Author      :   Usercyk
@Description :   Binary tree treasure
"""
from typing import List


class Solution:
    """
    The solution class
    """

    def dp(self, tree: List[int], n: int, idx: int) -> int:
        """
        DP function to find the maximum treasure
        """
        if idx > n:
            return 0
        l = self.dp(tree, n, idx*2)
        r = self.dp(tree, n, idx*2+1)
        ll = self.dp(tree, n, idx*4)
        lr = self.dp(tree, n, idx*4+1)
        rl = self.dp(tree, n, idx*4+2)
        rr = self.dp(tree, n, idx*4+3)
        return max(tree[idx]+ll+lr+rl+rr, l+r)

    def solve(self) -> None:
        """
        Solve the problem
        """
        n = int(input())
        tree = [0, *map(int, input().split())]
        result = self.dp(tree, n, 1)
        print(result)


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250516203705.png)

## 2. 学习总结和收获

感觉这次月考还算简单🤔前面几道题基本上都是实现
