# coding: utf-8
"""
@File        :   reverse_list_206.py
@Time        :   2025/03/17 14:36:15
@Author      :   Usercyk
@Description :   Reverse list
"""
from typing import Optional


class ListNode:
    """
    The list node
    """

    def __init__(self, x):
        self.val = x
        self.next: Optional[ListNode] = None


class Solution:
    """
    The solution
    """

    def reverse_list_iter(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverst the list using iteration

        Arguments:
            head -- list

        Returns:
            reversed list
        """
        p1, p2 = None, None
        while head is not None:
            p2 = head.next
            head.next = p1
            p1 = head
            head = p2

        return p1

    def reverse_list_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverst the list recursivly

        Arguments:
            head -- list

        Returns:
            reversed list
        """
        if head is None or head.next is None:
            return head

        tail = self.reverse_list_recursive(head.next)

        head.next.next = head
        head.next = None

        return tail
