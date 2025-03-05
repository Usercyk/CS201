# coding: utf-8
"""
@File        :   invert_index_06640.py
@Time        :   2025/03/01 16:56:07
@Author      :   Usercyk
@Description :   Realize the inverted index of the document
"""
from typing import Dict, List, Optional, Set


class Solution:
    """
    The Solution
    """

    def __init__(self) -> None:
        self.inverted_index: Dict[str, Set[int]] = {}

    def update(self, document_index: int, document: str) -> None:
        """
        Update the inverted index with the given document

        Arguments:
            document_index -- the index of the document
            document -- the content of the document
        """
        keywords = document.split()[1:]
        for keyword in keywords:
            cur: Set[int] = self.inverted_index.get(keyword, set())
            cur.add(document_index)
            self.inverted_index[keyword] = cur

    def query(self, keyword: str) -> Optional[List[int]]:
        """
        Query the index of the documents that contain the keyword

        Arguments:
            keyword -- The keyword

        Returns:
            The sorted index of the documents that contain the keyword
        """
        idx = self.inverted_index.get(keyword, None)
        if idx is not None:
            return sorted(idx)
        return None

    def solve(self) -> None:
        """
        Solve the problem
        """
        t = int(input())

        for i in range(t):
            self.update(i+1, input())
        q = int(input())

        for _ in range(q):
            res = self.query(input())
            if res is None:
                print("NOT FOUND")
            else:
                print(*res)


if __name__ == "__main__":
    Solution().solve()
