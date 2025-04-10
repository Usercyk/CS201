# coding: utf-8
"""
@File        :   covert_to_bst_108.py
@Time        :   2025/04/10 13:40:35
@Author      :   Usercyk
@Description :   Convert sorted array to binary search tree.
"""
from typing import List, Optional


class TreeNode:
    """
    The tree node
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val: int = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def sorted_array_to_bst(self, nums: List[int]) -> Optional[TreeNode]:
        """
        Convert sorted array to binary search tree.

        Arguments:
            nums -- The sorted array

        Returns:
            The binary search tree
        """
        if not nums:
            return None

        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sorted_array_to_bst(nums[:mid])
        root.right = self.sorted_array_to_bst(nums[mid + 1:])

        return root
