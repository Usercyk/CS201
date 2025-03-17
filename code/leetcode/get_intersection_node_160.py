# coding: utf-8
"""
@File        :   get_intersection_node_160.py
@Time        :   2025/03/17 14:15:41
@Author      :   Usercyk
@Description :   Get the intersection node of two LinkNode
"""
from typing import Optional


class ListNode:
    """
    The list node
    """

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    The solution
    """

    def get_intersection_node(self, head_a: ListNode, head_b: ListNode) -> Optional[ListNode]:
        """
        Get the intersection node of two single-linked list

        Arguments:
            head_a -- one single-linked list
            head_b -- another single-linked list

        Returns:
            The intersection node
        """
        pointer_a, pointer_b = head_a, head_b

        while pointer_a != pointer_b:
            pointer_a = head_b if (pointer_a is None) else pointer_a.next
            pointer_b = head_a if (pointer_b is None) else pointer_b.next

        return pointer_a
