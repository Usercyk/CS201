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
