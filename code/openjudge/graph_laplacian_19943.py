# coding: utf-8
"""
@File        :   graph_laplacian_19943.py
@Time        :   2025/04/26 18:06:09
@Author      :   Usercyk
@Description :   Calculate the Laplacian matrix of a graph.
"""
from itertools import product


class Vertex:
    """
    The vertex in a graph.
    """

    def __init__(self, tag: int):
        self.tag = tag
        self.degree = 0

    def connect(self, other):
        """
        Connect this vertex to another vertex.

        Arguments:
            other -- The other vertex to connect to.
        """
        if not isinstance(other, Vertex):
            raise TypeError("other must be a Vertex instance")
        self.degree += 1
        other.degree += 1


class Graph:
    """
    A graph
    """

    def __init__(self, n: int):
        self.n = n
        self.vertices = {k: Vertex(k) for k in range(n)}
        self.adjacency_matrix = [[0]*n for _ in range(n)]

    def update(self, u: int, v: int):
        """
        Update the graph by connecting two vertices.

        Arguments:
            u -- The first vertex.
            v -- The second vertex.
        """
        if u not in self.vertices or v not in self.vertices:
            raise ValueError("u and v must be in the graph")
        self.vertices[u].connect(self.vertices[v])
        self.adjacency_matrix[u][v] = 1
        self.adjacency_matrix[v][u] = 1

    @property
    def laplacian_matrix(self):
        """
        Calculate the Laplacian matrix of the graph.

        Returns:
            The Laplacian matrix of the graph.
        """
        laplacian = [[0]*self.n for _ in range(self.n)]
        for i, j in product(range(self.n), repeat=2):
            if i == j:
                laplacian[i][j] = self.vertices[i].degree
            else:
                laplacian[i][j] = -self.adjacency_matrix[i][j]
        return laplacian


if __name__ == "__main__":
    N, M = map(int, input().split())
    graph = Graph(N)
    for _ in range(M):
        U, V = map(int, input().split())
        graph.update(U, V)
    for row in graph.laplacian_matrix:
        print(*row)
