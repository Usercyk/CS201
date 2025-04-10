# coding: utf-8
"""
@File        :   bracket_to_pre_post_order_24729.py
@Time        :   2025/04/10 15:43:10
@Author      :   Usercyk
@Description :   Convert a bracket expression to pre-order and post-order traversals.
"""
from typing import Optional


class TreeNode:
    """
    Tree node
    """

    def __init__(self, val: str):
        self.val = val
        self.children = []

    def add_child(self, child: 'TreeNode'):
        """
        Add a child to the tree node

        Arguments:
            child -- The child node to be added
        """
        self.children.append(child)


class Solution:
    """
    The solution class
    """

    def build_tree(self, bracket_expression: str):
        """
        Build a tree from a bracket expression.

        Arguments:
            bracket_expression -- The bracket expression to be converted.

        Examples:
            A(B(E),C(F,G),D(H(I)))
        """
        root = None
        stack = []
        current = None
        for c in bracket_expression:
            if c == "(":
                if current is not None:
                    stack.append(current)
            elif c == ")":
                if stack:
                    current = stack.pop()
            elif c == ',':
                if stack:
                    current = stack[-1]
            elif c.isalpha():
                node = TreeNode(c)
                if root is None:
                    root = node
                if current is not None:
                    current.add_child(node)
                current = node
        return root

    def pre_order(self, root: Optional[TreeNode]) -> str:
        """
        Pre-order traversal of the tree.

        Arguments:
            root -- The root of the tree.

        Returns:
            The pre-order traversal of the tree.
        """
        if root is None:
            return ""
        res = root.val
        for child in root.children:
            res += self.pre_order(child)
        return res

    def post_order(self, root: Optional[TreeNode]) -> str:
        """
        Post-order traversal of the tree.

        Arguments:
            root -- The root of the tree.

        Returns:
            The post-order traversal of the tree.
        """
        if root is None:
            return ""
        res = ""
        for child in root.children:
            res += self.post_order(child)
        res += root.val
        return res

    def solve(self) -> None:
        """
        Solve the problem
        """
        expr = input().strip()
        root = self.build_tree(expr)
        print(self.pre_order(root))
        print(self.post_order(root))


if __name__ == "__main__":
    Solution().solve()
