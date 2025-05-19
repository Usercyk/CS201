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
