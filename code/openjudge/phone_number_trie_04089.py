# coding: utf-8
"""
@File        :   phone_number_trie_04089.py
@Time        :   2025/04/26 18:56:34
@Author      :   Usercyk
@Description :   Build a trie to check if a number is a prefix of another number.
"""


from typing import Dict


class TrieNode:
    """
    Trie node
    """

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end = False


class Trie:
    """
    Trie
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, number: str) -> bool:
        """
        Insert a number into the trie.

        Arguments:
            number -- the number to insert

        Returns:
            True if the number is not a prefix of any existing number, False otherwise
        """
        node = self.root
        for digit in number:
            if digit not in node.children:
                node.children[digit] = TrieNode()
            node = node.children[digit]
            if node.is_end:
                return False
        node.is_end = True
        if node.children:
            return False
        return True


class Solution:
    """
    The solution class
    """

    def solve(self) -> None:
        """
        Solve the problem.
        """
        for _ in range(int(input())):
            n = int(input())
            trie = Trie()
            numbers = [input().strip() for _ in range(n)]
            flag = True
            for number in numbers:
                if not trie.insert(number):
                    flag = False
                    break
            print("YES" if flag else "NO")


if __name__ == "__main__":
    Solution().solve()
