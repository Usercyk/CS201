# coding: utf-8
"""
@File        :   adjust_score_22528.py
@Time        :   2025/05/03 00:15:08
@Author      :   Usercyk
@Description :   Adjust Score
"""
from bisect import bisect
from typing import List


class Solution:
    """
    The solution class
    """
    MAX_B = 1_000_000_000

    def is_ok(self, scores: List[float], n: int, b: int) -> bool:
        """
        Check if the score adjustment is valid
        """
        a = b/self.MAX_B
        ns = [a*x+1.1**(a*x) for x in scores]
        idx = bisect(ns, 85.0)
        return idx <= n*0.4

    def solve(self) -> None:
        """
        The main function to solve the problem
        """
        scores = sorted(map(float, input().split()))
        n = len(scores)
        l, r = 1, self.MAX_B
        while l < r:
            m = (l+r)//2
            if self.is_ok(scores, n, m):
                r = m
            else:
                l = m+1
        print(l)


if __name__ == '__main__':
    Solution().solve()
