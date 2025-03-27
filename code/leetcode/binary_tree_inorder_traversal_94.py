# coding: utf-8
"""
@File        :   binary_tree_inorder_traversal_94.py
@Time        :   2025/03/27 14:03:55
@Author      :   Usercyk
@Description :   Inorder traversal
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

    def inorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal using recursion

        Arguments:
            root -- The root node

        Returns:
            the inorder traversal
        """
        if root is None:
            return []
        return [*self.inorder_traversal_recursive(root.left),
                root.val,
                *self.inorder_traversal_recursive(root.right)]

    def inorder_traversal_iter(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal using iterating

        Arguments:
            root -- The root node

        Returns:
            the inorder traversal
        """
        res = []
        stack = []

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            assert root is not None
            res.append(root.val)
            root = root.right
        return res
