# coding: utf-8
"""
@File        :   single_number_136.py
@Time        :   2025/03/17 13:36:56
@Author      :   Usercyk
@Description :   Find the number only exists once
"""
from functools import reduce
from operator import xor
from typing import List


class Solution:
    "The solution"

    def single_number(self, nums: List[int]) -> int:
        """
        Exclusive all to find the single number

        Arguments:
            nums -- All numbers

        Returns:
            The single number
        """
        return reduce(xor, nums)
