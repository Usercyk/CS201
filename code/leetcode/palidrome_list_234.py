# coding: utf-8
"""
@File        :   palidrome_list_234.py
@Time        :   2025/03/23 23:29:05
@Author      :   Usercyk
@Description :   Check if a linked list is a palindrome
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

    def is_palindrome(self, head: Optional[ListNode]) -> bool:
        """
        Check if a linked list is a palindrome

        Arguments:
            head -- the head of the linked list

        Returns:
            whether it is a palindrome
        """

        if head is None:
            return True
        fast = head
        slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            assert slow is not None
            slow = slow.next

        assert slow is not None
        slow = self.reverse(slow.next)

        while slow is not None:
            assert head is not None
            if head.val != slow.val:
                return False
            head = head.next
            slow = slow.next

        return True

    def reverse(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverse the linked list

        Arguments:
            head -- the head of the linked list

        Returns:
            the reversed linked list
        """
        prev = None
        while head:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev
