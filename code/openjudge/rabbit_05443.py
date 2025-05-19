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
