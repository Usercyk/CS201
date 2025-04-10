# coding: utf-8
"""
@File        :   minimum_pair_removal_to_sort_3510.py
@Time        :   2025/04/10 17:04:59
@Author      :   Usercyk
@Description :   Minimum Pair Removal to Sort
"""
from itertools import pairwise
from typing import List

from sortedcontainers import SortedList


class Solution:
    """
    The solutino class
    """

    def minimum_pair_removal(self, nums: List[int]) -> int:
        """
        Calculate the minimum pair removal to sort

        Arguments:
            nums -- The list of integers

        Returns:
            The minimum pair removal to sort
        """
        pq = SortedList()
        idxs = SortedList(range(len(nums)))
        dec = 0

        for i, (x, y) in enumerate(pairwise(nums)):
            if x > y:
                dec += 1
            pq.add((x+y, i))

        cnt = 0
        while dec:
            cnt += 1
            s, i = pq.pop(0)
            k = idxs.bisect_left(i)

            nxt_i: int = idxs[k+1]  # type: ignore
            dec -= nums[i] > nums[nxt_i]

            if k:
                pre_i: int = idxs[k-1]  # type: ignore
                dec = dec-(nums[pre_i] > nums[i])+(nums[pre_i] > s)

                pq.remove((nums[pre_i]+nums[i], pre_i))
                pq.add((s+nums[pre_i], pre_i))

            if k+2 < len(idxs):
                nxt_nxt_i: int = idxs[k+2]  # type: ignore
                dec = dec-(nums[nxt_i] > nums[nxt_nxt_i])+(s > nums[nxt_nxt_i])

                pq.remove((nums[nxt_i]+nums[nxt_nxt_i], nxt_i))
                pq.add((s+nums[nxt_nxt_i], i))

            nums[i] = s
            idxs.remove(nxt_i)

        return cnt
