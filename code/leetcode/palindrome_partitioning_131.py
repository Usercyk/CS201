# coding: utf-8
"""
@File        :   palindrome_partitioning_131.py
@Time        :   2025/03/27 15:18:45
@Author      :   Usercyk
@Description :   Palindrome partitioning
"""
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.s: Optional[str] = None
        self.n: int = 0
        self.palindromes: Optional[List[List[bool]]] = None
        self.partitionings: List[List[str]] = []
        self.dfs_temp: List[str] = []

    def get_all_palindrome(self) -> List[List[bool]]:
        """
        Get all palindrome

        Returns:
            if s[i:j+1] is palindrome
        """
        assert self.s is not None
        palindromes = [[True]*self.n for _ in range(self.n)]
        for i in range(self.n-1, -1, -1):
            for j in range(i+1, self.n):
                palindromes[i][j] = (self.s[i] == self.s[j]
                                     ) and palindromes[i+1][j-1]
        return palindromes

    def dfs(self, idx: int = 0) -> None:
        """
        Dfs the string

        Keyword Arguments:
            idx -- the current index (default: {0})
        """
        assert self.palindromes is not None
        assert self.s is not None
        if idx >= self.n:
            self.partitionings.append(self.dfs_temp.copy())
            return
        for j in range(idx, self.n):
            if self.palindromes[idx][j]:
                self.dfs_temp.append(self.s[idx:j+1])
                self.dfs(j+1)
                self.dfs_temp.pop()

    def partition(self, s: str) -> List[List[str]]:
        """
        Palindrome partitioning

        Arguments:
            s -- the string

        Returns:
            all possible partitioning
        """
        self.s = s
        self.n = len(s)
        self.palindromes = self.get_all_palindrome()
        self.dfs()
        return self.partitionings
