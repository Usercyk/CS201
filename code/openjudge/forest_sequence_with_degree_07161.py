# coding: utf-8
"""
@File        :   forest_sequence_with_degree_07161.py
@Time        :   2025/04/06 16:04:02
@Author      :   Usercyk
@Description :   Forest hierarchical sequence storage with degrees
"""

from collections import deque
from typing import List, Optional


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
    The solution
    """

    def build_tree(self, sequence: list) -> Optional[TreeNode]:
        """
        Build the tree from the sequence

        Arguments:
            sequence -- The sequence of the tree

        Returns:
            The root node of the tree

        Example:
            sequence: C 3 E 3 F 0 G 0 K 0 H 0 J 0
            results:
                C
                ├── E
                │   ├── K
                │   ├── H
                │   └── J
                ├── F
                ├── G
        """
        if not sequence:
            return None

        root = TreeNode(sequence[0])
        q = deque([(root, int(sequence[1]))])
        i = 2

        while i < len(sequence):
            node, degree = q.popleft()
            for _ in range(degree):
                child = TreeNode(sequence[i])
                node.add_child(child)
                q.append((child, int(sequence[i + 1])))
                i += 2

        return root

    def post_order_traversal(self, node: Optional[TreeNode]) -> List[str]:
        """
        Post-order traversal of the tree

        Arguments:
            node -- The root node of the tree

        Returns:
            The post-order traversal of the tree
        """
        if node is None:
            return []

        result = []
        for child in node.children:
            result.extend(self.post_order_traversal(child))

        result.append(node.val)
        return result

    def solve(self) -> None:
        """
        Solve the problem
        """
        n = int(input())
        for _ in range(n):
            sequence = input().split()
            root = self.build_tree(sequence)
            result = self.post_order_traversal(root)
            print(' '.join(result), end=' ')


if __name__ == "__main__":
    Solution().solve()
