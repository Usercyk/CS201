# coding: utf-8
"""
@File        :   topological_sort_04084.py
@Time        :   2025/05/16 19:42:12
@Author      :   Usercyk
@Description :   Topological Sort
"""
from typing import List


class Vertex:
    """
    The vertex
    """

    def __init__(self, tag: int) -> None:
        self.tag: int = tag
        self.in_degree: int = 0
        self.out_degree: int = 0
        self.out_adjacent: List["Vertex"] = []

    def __str__(self) -> str:
        return f"v{self.tag}"

    def connect_to(self, other: "Vertex"):
        """
        Connect this vertex to another vertex

        Arguments:
            other -- The other vertex

        Returns:
            The other vertex
        """
        self.out_adjacent.append(other)
        self.out_degree += 1
        other.in_degree += 1

    def __lt__(self, other: "Vertex") -> bool:
        """
        Compare two vertices

        Arguments:
            other -- The other vertex

        Returns:
            True if this vertex is less than the other vertex
        """
        return self.tag < other.tag


class Graph:
    """
    The graph
    """

    def __init__(self, n: int) -> None:
        self._vertices: List[Vertex] = [Vertex(i+1) for i in range(n)]

    def connect(self, u: int, v: int) -> None:
        """
        Connect u to v

        Arguments:
            u -- The first vertex
            v -- The second vertex
        """
        self._vertices[u].connect_to(self._vertices[v])

    def topological_sort(self) -> List[str]:
        """
        Topological sort the graph

        Returns:
            The topological order of the graph
        """
        order: List[str] = []
        queue: List[Vertex] = [v for v in self._vertices if v.in_degree == 0]
        while queue:
            vertex = queue.pop(0)
            order.append(str(vertex))
            for adj in vertex.out_adjacent:
                adj.in_degree -= 1
                if adj.in_degree == 0:
                    queue.append(adj)
            queue.sort()
        return order


class Solution:
    """
    The solution class
    """

    def solve(self) -> None:
        """
        Solve the problem
        """
        v, a = map(int, input().split())
        graph = Graph(v)
        for _ in range(a):
            u, v = map(int, input().split())
            graph.connect(u-1, v-1)
        order = graph.topological_sort()
        print(*order)


if __name__ == "__main__":
    Solution().solve()
