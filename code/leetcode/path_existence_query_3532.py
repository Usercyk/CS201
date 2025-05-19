# coding: utf-8
"""
@File        :   path_existence_query_3532.py
@Time        :   2025/05/03 00:04:25
@Author      :   Usercyk
@Description :   Path existence query
"""
from typing import List


class Solution:
    """
    The solution class
    """

    # pylint: disable=C0103
    def pathExistenceQueries(self,
                             n: int,
                             nums: List[int],
                             maxDiff: int,
                             queries: List[List[int]]) -> List[bool]:
        """
        Check if there exists a path between two indices in the array nums.

        Arguments:
            n -- number of elements in nums
            nums -- the array of integers
            maxDiff -- the maximum allowed difference between adjacent elements
            queries -- the list of queries, each query is a list of two integers [u, v]

        Returns:
            A list of boolean values indicating if a path exists for each query.
        """
        tag = [0]*n
        for i in range(1, n):
            tag[i] = tag[i-1] + int(abs(nums[i]-nums[i-1]) > maxDiff)
        return [tag[u] == tag[v] for u, v in queries]
