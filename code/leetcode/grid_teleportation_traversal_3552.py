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
                for tx, ty in tele[matrix[x][y]]:
                    if dist <= distance[tx][ty]:
                        distance[tx][ty] = dist
                        q.appendleft((tx, ty))
                del tele[matrix[x][y]]

        return -1
