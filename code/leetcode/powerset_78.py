# coding: utf-8
"""
@File        :   powerset_78.py
@Time        :   2025/04/26 18:16:39
@Author      :   Usercyk
@Description :   Find the power set of a given set of numbers.
"""
from itertools import compress
from typing import List


class Solution:
    """
    The solution class
    """

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Find the power set of a given set of numbers.

        Arguments:
            nums -- A list of integers representing the set.

        Returns:
            A list of lists representing the power set.
        """
        return [list(compress(nums, (int(c) for c in format(i, f"0{len(nums)}b"))))
                for i in range(1 << len(nums))]
