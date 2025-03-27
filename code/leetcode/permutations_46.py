# coding: utf-8
"""
@File        :   permutations_46.py
@Time        :   2025/03/27 13:08:25
@Author      :   Usercyk
@Description :   Permute
"""


from itertools import permutations
from typing import List


class Solution:
    """
    The solution class
    """

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Permute the nums

        Arguments:
            nums -- Numbers

        Returns:
            Permutations
        """
        return [list(t) for t in permutations(nums)]
