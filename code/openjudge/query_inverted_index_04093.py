# coding: utf-8
"""
@File        :   query_inverted_index_04093.py
@Time        :   2025/03/01 17:21:07
@Author      :   Usercyk
@Description :   Query the inverted index of the document
"""

from typing import List, Optional, Set


class Solution:
    """
    The Solution
    """

    def __init__(self) -> None:
        self.inverted_index: List[Set[int]] = []

    def update(self, idxs: str) -> None:
        """
        Update the inverted index with the given indices

        Arguments:
            idxs -- The indices
        """
        self.inverted_index.append(set(map(int, idxs.split()[1:])))

    def query(self, requirement: str) -> Optional[Set[int]]:
        """
        Query the indices of the documents that fit the requirement

        Arguments:
            requirement -- The requirement

        Returns:
            The indices of the documents that fit the requirement
        """
        req = map(int, requirement.split())
        wait = []
        res: Optional[Set[int]] = None

        for i, r in enumerate(req):
            if r == 0:
                continue
            if r == 1:
                if res is None:
                    res = self.inverted_index[i]
                else:
                    res = res.intersection(self.inverted_index[i])
            elif r == -1:
                if res is None:
                    wait.append(i)
                else:
                    res = res.difference(self.inverted_index[i])

        for w in wait:
            if res is not None:
                res = res.difference(self.inverted_index[w])

        return res

    def solve(self):
        """
        Solve the problem
        """
        t = int(input())
        for _ in range(t):
            self.update(input())
        q = int(input())
        for _ in range(q):
            res = self.query(input())
            if res is None or len(res) == 0:
                print("NOT FOUND")
            else:
                print(*sorted(res))


if __name__ == "__main__":
    Solution().solve()
