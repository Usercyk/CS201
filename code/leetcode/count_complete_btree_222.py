# coding: utf-8
"""
@File        :   count_complete_btree_222.py
@Time        :   2025/04/21 19:16:52
@Author      :   Usercyk
@Description :   Complete Binary Tree Node Count
"""
from typing import Optional


class TreeNode:
    """
    Binary tree
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    The solution class
    """

    def find(self, root: Optional[TreeNode], level: int, tag: int) -> bool:
        """
        Find if the corresponding node exists in the complete binary tree.

        Arguments:
            root -- The root of the binary tree
            level -- The level of the binary tree
            tag -- The tag of the binary tree

        Returns:
            True if the node exists, False otherwise
        """
        bits = 1 << (level-1)
        node = root
        while node is not None and bits > 0:
            if bits & tag:
                node = node.right
            else:
                node = node.left
            bits >>= 1

        return node is not None

    def count_nodes(self, root: Optional[TreeNode]) -> int:
        """
        Count the number of nodes in a complete binary tree.

        Arguments:
            root -- The root of the binary tree

        Returns:
            The number of nodes in the complete binary tree
        """
        if root is None:
            return 0

        h = 0
        node = root
        while node is not None:
            h += 1
            node = node.left

        l = 1 << (h-1)
        r = (1 << h)-1
        while l < r:
            mid = (l+r+1)//2
            if self.find(root, h-1, mid):
                l = mid
            else:
                r = mid-1

        return l
