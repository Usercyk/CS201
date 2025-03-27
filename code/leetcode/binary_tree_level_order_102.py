# coding: utf-8
"""
@File        :   binary_tree_level_order_102.py
@Time        :   2025/03/27 14:46:09
@Author      :   Usercyk
@Description :   Level order
"""
from typing import List, Optional


class TreeNode:
    """
    The binary tree
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Level order

        Arguments:
            root -- The root node

        Returns:
            level order
        """
        if root is None:
            return []

        res = []
        queue = [root]

        temp_res = []
        temp_queue = []

        while queue:
            for node in queue:
                temp_res.append(node.val)
                if node.left is not None:
                    temp_queue.append(node.left)
                if node.right is not None:
                    temp_queue.append(node.right)
            queue = temp_queue
            res.append(temp_res)
            temp_res = []
            temp_queue = []

        return res
