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
