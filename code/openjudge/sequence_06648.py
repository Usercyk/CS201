# coding: utf-8
"""
@File        :   sequence_06648.py
@Time        :   2025/03/07 19:56:35
@Author      :   Usercyk
@Description :   Merge Sequence
"""
import heapq
from typing import List


class Solution:
    """
    The solution class
    """

    def __init__(self):
        self.n = 0

    def merge(self, a: List[int], b: List[int]) -> List[int]:
        """
        Merge the sorted array to get first n minimal sum

        Arguments:
            a -- a sorted array
            b -- another sorted array

        Returns:
            first n minimal sum
        """
        heap = []
        visited = set()

        heapq.heappush(heap, (a[0]+b[0], 0, 0))
        visited.add((0, 0))

        res = []
        len_a = len(a)
        len_b = len(b)

        while heap:
            if len(res) == self.n:
                break
            s, i, j = heapq.heappop(heap)
            res.append(s)
            if i+1 < len_a and (i+1, j) not in visited:
                heapq.heappush(heap, (a[i+1]+b[j], i+1, j))
                visited.add((i+1, j))
            if j+1 < len_b and (i, j+1) not in visited:
                heapq.heappush(heap, (a[i]+b[j+1], i, j+1))
                visited.add((i, j+1))

        return res

    def solve(self):
        """
        Solve the problem
        """
        for _ in range(int(input())):
            m, self.n = map(int, input().split())
            arr = [sorted(map(int, input().split())) for _ in range(m)]
            ans = arr.pop()
            while arr:
                ans = self.merge(ans, arr.pop())
            print(*ans)


if __name__ == "__main__":
    Solution().solve()
