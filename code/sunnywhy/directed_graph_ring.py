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
