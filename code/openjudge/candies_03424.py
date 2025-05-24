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
