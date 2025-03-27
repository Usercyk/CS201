# coding: utf-8
"""
@File        :   word_search_79.py
@Time        :   2025/03/27 13:20:38
@Author      :   Usercyk
@Description :   Search word with backtracking
"""
from itertools import product
from typing import List, Optional, Tuple


class Solution:
    """
    The solution class
    """
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self) -> None:
        self.board: Optional[List[List[str]]] = None
        self.m: int = 0
        self.n: int = 0
        self.word: Optional[str] = None
        self.visited: Optional[List[List[bool]]] = None

    def search(self, board_pos: Tuple[int, int], word_pos: int) -> bool:
        """
        Match the word from word_pos with the board from board_pos

        Arguments:
            board_pos -- current board pos
            word_pos -- current word pos

        Returns:
            Whether the left part can be matched
        """
        if self.board is None or self.word is None or self.visited is None:
            return False
        x, y = board_pos

        if word_pos >= len(self.word):
            return True
        if self.board[x][y] != self.word[word_pos]:
            return False

        if word_pos == len(self.word)-1:
            return True

        self.visited[x][y] = True
        nw = word_pos+1
        for dx, dy in self.DIRECTIONS:
            nx, ny = x+dx, y+dy
            if nx < 0 or nx >= self.m or ny < 0 or ny >= self.n or self.visited[nx][ny]:
                continue
            if self.search((nx, ny), nw):
                return True

        self.visited[x][y] = False
        return False

    def reset_visited(self) -> None:
        """
        Reset the visited according to the board
        """
        if self.board is None or not self.board[0]:
            return None
        self.visited = [[False]*self.n for _ in range(self.m)]

    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        Check if the word exists

        Arguments:
            board -- The character board
            word -- The word used to be found

        Returns:
            The research result
        """
        self.board = board
        self.word = word
        self.m = len(self.board)
        self.n = len(self.board[0])
        self.reset_visited()
        for pos in product(range(self.m), range(self.n)):
            if self.search(pos, 0):
                return True
        return False


print(Solution().exist([["a"]], "a"))
