# coding: utf-8
"""
@File        :   traverse_27928.py
@Time        :   2025/04/10 13:56:58
@Author      :   Usercyk
@Description :   Traverse the tree in the "ascending" order.
"""
from heapq import heappop, heappush
from typing import Optional


class TreeNode:
    """
    Tree node
    """

    def __init__(self, val: int):
        self.val = val
        self.left_children = []
        self.right_children = []

    def add_child(self, child: 'TreeNode'):
        """
        Add a child to the tree node

        Arguments:
            child -- The child node to be added
        """
        if child < self:
            heappush(self.left_children, child)
        else:
            heappush(self.right_children, child)

    def __lt__(self, other: 'TreeNode') -> bool:
        """
        Compare two tree nodes

        Arguments:
            other -- The other tree node to be compared

        Returns:
            True if the current node is less than the other node, False otherwise
        """
        return self.val < other.val


class Solution:
    """
    The solution class
    """

    def __init__(self):
        self.nodes = {}

    def get_node(self, val: int, as_child: bool = False) -> TreeNode:
        """
        Get the node with the given value

        Arguments:
            val -- The value of the node

        Returns:
            The node with the given value
        """
        if val not in self.nodes:
            self.nodes[val] = (TreeNode(val), as_child)
        node, is_child = self.nodes[val]
        self.nodes[val] = (node, is_child or as_child)
        return node

    def build_tree(self) -> Optional[TreeNode]:
        """
        Build the tree

        Returns:
            The root of the tree
        """
        n = int(input())
        for _ in range(n):
            *a, = map(int, input().split())
            p = self.get_node(a.pop(0))
            for i in a:
                c = self.get_node(i, True)
                p.add_child(c)

        root = None

        for node, is_child in self.nodes.values():
            if not is_child:
                root = node
                break

        return root

    def traverse(self, root: Optional[TreeNode]) -> None:
        """
        Traverse the tree in the "ascending" order
        """
        if root is None:
            return
        while root.left_children:
            self.traverse(heappop(root.left_children))
        print(root.val)
        while root.right_children:
            self.traverse(heappop(root.right_children))


if __name__ == '__main__':
    sol = Solution()
    sol.traverse(sol.build_tree())
