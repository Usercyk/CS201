# coding: utf-8
"""
@File        :   fajo_04135.py
@Time        :   2025/02/23 18:38:42
@Author      :   Usercyk
@Description :   Minimum value of the maximum cost of fajo months
"""
from functools import reduce
from typing import Callable, List


class Solution:
    """
    The solution
    """

    @staticmethod
    def bisect(a, x, key):
        """
        Copy of the 3.13.2 bisect.bisect
        """
        lo = 0
        hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def bisect_key(self, m: int, costs: List[int]) -> Callable[[int], int]:
        """
        Generate the bisect key from the costs

        Arguments:
            m -- The maximum number of the fajo months
            costs -- The cost of every day

        Returns:
            The key function for bisect
        """
        def func(previous, cur):
            pre_fajo, pre_cost, x = previous
            if cur+pre_cost > x:
                return (pre_fajo+1, cur, x)
            return (pre_fajo, pre_cost+cur, x)

        return lambda x: m-reduce(func, costs, (0, 0, x))[0]

    def solve(self, m: int, costs: List[int]) -> int:
        """
        Solve the minimum value of the maximum cost of fajo months

        Arguments:
            m -- The maximum number of the fajo months
            costs -- The cost of every day

        Returns:
            The minimum value of the maximum cost of fajo months
        """
        ran = range(max(costs), sum(costs)+1)
        return ran[self.bisect(ran, 0, key=self.bisect_key(m, costs))]


if __name__ == "__main__":
    N, M = map(int, input().split())
    Costs = [int(input()) for _ in range(N)]
    print(Solution().solve(M, Costs))
