# coding: utf-8
"""
@File        :   level_order_of_bst_05455.py
@Time        :   2025/04/21 20:22:10
@Author      :   Usercyk
@Description :   Build a binary search tree and print its level order traversal.
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

    def level_order(self, root: Optional[TreeNode]) -> List[int]:
        """
        Perform a level order traversal of the binary search tree.

        Arguments:
            root -- the root of the tree

        Returns:
            A list of values in level order
        """
        if not root:
            return []

        queue = [root]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def build(self, array: List[int]) -> Optional[TreeNode]:
        """
        Build a binary search tree from the given array.

        Arguments:
            array -- the array to build the tree from

        Returns:
            The root of the binary search tree
        """
        if not array:
            return None

        root = TreeNode(array[0])
        for i in range(1, len(array)):
            self.insert(root, array[i])
        return root

    def insert(self, root: TreeNode, val: int) -> None:
        """
        Insert a value into the binary search tree.

        Arguments:
            root -- the root of the tree
            val -- the value to insert
        """
        if val == root.val:
            return
        if val < root.val:
            if root.left is None:
                root.left = TreeNode(val)
            else:
                self.insert(root.left, val)
        else:
            if root.right is None:
                root.right = TreeNode(val)
            else:
                self.insert(root.right, val)


if __name__ == "__main__":
    sol = Solution()
    rt = sol.build(list(map(int, input().split())))
    res = sol.level_order(rt)
    print(*res)
