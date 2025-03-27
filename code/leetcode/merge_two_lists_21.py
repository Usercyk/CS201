# coding: utf-8
"""
@File        :   merge_two_lists_21.py
@Time        :   2025/03/23 23:21:10
@Author      :   Usercyk
@Description :   Merge two sorted lists
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

    def merge_two_lists(self,
                        list1: Optional[ListNode],
                        list2: Optional[ListNode]
                        ) -> Optional[ListNode]:
        """
        Merge two sorted lists

        Arguments:
            list1 -- list 1
            list2 -- list 2

        Returns:
            merged list
        """
        if list1 is None:
            return list2
        if list2 is None:
            return list1

        if list1.val < list2.val:
            list1.next = self.merge_two_lists(list1.next, list2)
            return list1

        list2.next = self.merge_two_lists(list1, list2.next)
        return list2
