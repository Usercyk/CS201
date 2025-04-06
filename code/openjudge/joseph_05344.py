# coding: utf-8
"""
@File        :   joseph_05344.py
@Time        :   2025/04/06 15:44:39
@Author      :   Usercyk
@Description :   Josephus Problem
"""
from collections import deque
from typing import List


class Solution:
    """
    The solution class
    """

    def josephus(self, n: int, k: int) -> List[int]:
        """
        Josephus problem

        Arguments:
            n -- number of people in the circle
            k -- step count

        Returns:
            The order of people being eliminated
        """
        res = []
        people = deque(range(1, n + 1))
        while len(people) > 1:
            people.rotate(-(k - 1))
            res.append(people.popleft())

        return res


if __name__ == "__main__":
    N, K = map(int, input().split())
    result = Solution().josephus(N, K)
    print(*result)
