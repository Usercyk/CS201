# coding: utf-8
"""
@File        :   minimum_size_1760.py
@Time        :   2025/02/23 17:45:19
@Author      :   Usercyk
@Description :   Bisect to calculate the minimum size
"""
from bisect import bisect
from typing import Callable, List


class Solution:
    """
    The solution
    """

    def bisect_key(self, nums: List[int], max_operations: int) -> Callable[[int], int]:
        """
        Generate the bisect key from the numbers

        Arguments:
            nums -- the array
            max_operations -- the possible operation times

        Returns:
            The key function for bisect
        """
        return lambda x: max_operations-sum((num-1)//x for num in nums)+1

    def minimum_size(self, nums: List[int], max_operations: int) -> int:
        """
        Calculate the minimum size of the largest value in the array after operation

        Arguments:
            nums -- the array
            max_operations -- the possible operation times

        Returns:
            The minimum size of the maximum value after operation
        """
        return bisect(range(1, max(nums)+1), 0, key=self.bisect_key(nums, max_operations))+1


if __name__ == "__main__":
    print(Solution().minimum_size([7, 17], 2))
