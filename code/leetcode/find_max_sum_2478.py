# coding: utf-8
"""
@File        :   find_max_sum_2478.py
@Time        :   2025/03/17 14:47:57
@Author      :   Usercyk
@Description :   Find the max sum of k value
"""
from heapq import heappush, heappushpop
from typing import List


class Solution:
    """
    The solution
    """

    def find_max_sum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        Find the max sum

        Arguments:
            nums1 -- The first array
            nums2 -- The second array
            k -- k

        Returns:
            The max k sum
        """
        st = sorted((val, nums2[idx], idx)
                    for idx, val in enumerate(nums1))

        heap = []
        len_heap = 0

        res = 0
        ans = [0]*len(nums1)

        for i, (n1, n2, idx) in enumerate(st):
            if i > 0 and n1 == st[i-1][0]:
                ans[idx] = ans[st[i-1][2]]  # type: ignore
            else:
                ans[idx] = res
            res += n2

            if len_heap < k:
                heappush(heap, n2)
                len_heap += 1
            else:
                res -= heappushpop(heap, n2)

        return ans
