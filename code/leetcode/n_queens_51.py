# coding: utf-8
"""
@File        :   n_queens_51.py
@Time        :   2025/04/26 19:07:05
@Author      :   Usercyk
@Description :   Solve N queens problem
"""

from typing import List


class Solution:
    """
    The solution class
    """

    def backtrack(self,
                  n: int,
                  row: int,
                  col: int,
                  left_diagonal: int,
                  right_diagonal: int,
                  board: List[str],
                  result: List[List[str]]
                  ) -> None:
        """
        Backtrack to find all solutions

        Arguments:
            n -- the number of queens
            row -- the current row
            col -- the current column
            left_diagonal -- the left diagonal
            right_diagonal -- the right diagonal
            board -- the current board
            result -- the list of solutions
        """
        if row == n:
            result.append(board[:])
            return

        for i in range(n):
            if all(((col & (1 << i)) == 0,
                   (left_diagonal & (1 << (row + i))) == 0,
                   (right_diagonal & (1 << (row - i + n - 1))) == 0)):
                board[row] = '.' * i + 'Q' + '.' * (n - i - 1)
                self.backtrack(n, row + 1, col | (1 << i), left_diagonal | (1 << (row + i)),
                               right_diagonal | (1 << (row - i + n - 1)), board, result)
                board[row] = '.' * n

    def solve_n_queens(self, n: int) -> List[List[str]]:
        """
        Solve N queens

        Arguments:
            n -- the number of queens

        Returns:
            A list of solutions, each solution is a list of strings
        """
        result = []
        board = ['.' * n for _ in range(n)]
        self.backtrack(n, 0, 0, 0, 0, board, result)
        return result
