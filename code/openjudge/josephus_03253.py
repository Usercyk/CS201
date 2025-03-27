# coding: utf-8
"""
@File        :   josephus_03253.py
@Time        :   2025/03/24 00:15:39
@Author      :   Usercyk
@Description :   Solve josephus problem
"""
from collections import deque
from typing import List


class Solution:
    """
    The solution class
    """

    def josephus(self, n: int, p: int, m: int) -> List[int]:
        """
        Solve josephus problem

        Arguments:
            n -- n kids
            p -- p start
            m -- m out

        Returns:
            The order of getting out
        """
        queue = deque(range(1, n+1))
        queue.rotate(1-p)

        result = []

        while queue:
            for _ in range(m-1):
                queue.append(queue.popleft())
            result.append(queue.popleft())

        return result

    def solve(self):
        """
        Solve the problem
        """
        while True:
            n, p, m = map(int, input().split())
            if n == p == m == 0:
                break
            print(*self.josephus(n, p, m), sep=",")


if __name__ == "__main__":
    Solution().solve()
