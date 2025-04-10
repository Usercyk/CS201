# coding: utf-8
"""
@File        :   sum_root_to_leaf_numbers_129.py
@Time        :   2025/04/10 15:14:38
@Author      :   Usercyk
@Description :   Calculate the sum of all root-to-leaf numbers in a binary tree.
"""
from typing import Optional


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

    def sum_numbers(self, root: Optional[TreeNode]) -> int:
        """
        Calculate the sum of all root-to-leaf numbers in a binary tree.

        Arguments:
            root -- The root node of the binary tree

        Returns:
            The sum of all root-to-leaf numbers
        """
        return self.dfs(root, 0)

    def dfs(self, root: Optional[TreeNode], current: int) -> int:
        """
        Depth-first search to calculate the sum of all root-to-leaf numbers.

        Arguments:
            root -- The current node
            current -- The current number formed by the path from the root to this node

        Returns:
            The sum of all root-to-leaf numbers from this node
        """
        if root is None:
            return 0

        current = current * 10 + root.val

        if root.left or root.right:
            return self.dfs(root.left, current) + self.dfs(root.right, current)

        return current
