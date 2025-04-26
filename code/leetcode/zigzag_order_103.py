# coding: utf-8
"""
@File        :   zigzag_order_103.py
@Time        :   2025/04/21 19:52:12
@Author      :   Usercyk
@Description :   Zigzag Level Order Traversal of a Binary Tree
"""
from typing import List, Optional


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

    def zigzag_level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Zigzag level order traversal of a binary tree.

        Arguments:
            root -- The root of the binary tree

        Returns:
            A list of lists of integers representing the zigzag level order traversal
        """
        if not root:
            return []

        ans = []
        queue = [root]
        left_to_right = True

        while queue:
            nq = []
            vals = []
            for node in queue:
                vals.append(node.val)
                if node.left is not None:
                    nq.append(node.left)
                if node.right is not None:
                    nq.append(node.right)
            if not left_to_right:
                vals.reverse()
            ans.append(vals)
            queue = nq
            left_to_right = not left_to_right
        return ans
