# Assignment #A: Graph starts

Updated GMT+8 Apr 26, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：实现

代码：

```python
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

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426181456.png)

### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：Itertools启动！（bushi

根据集合论，S的幂集也就是S到{0, 1}的所有映射的集合，直接用N位二进制数来枚举即可。

而把已知二进制数转为nums中的几个数，可以使用itertools中的compress来提取

顺便进行一波压行（雾

代码：

```python
# coding: utf-8
"""
@File        :   powerset_78.py
@Time        :   2025/04/26 18:16:39
@Author      :   Usercyk
@Description :   Find the power set of a given set of numbers.
"""
from itertools import compress
from typing import List


class Solution:
    """
    The solution class
    """

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Find the power set of a given set of numbers.

        Arguments:
            nums -- A list of integers representing the set.

        Returns:
            A list of lists representing the power set.
        """
        # n = len(nums)
        # res = []
        # for i in range(1 << n):
        #     s = format(i, f"0{n}b")
        #     picker = [int(c) for c in s]
        #     res.append(list(compress(nums, picker)))
        # return res

        return [list(compress(nums, (int(c) for c in format(i, f"0{len(nums)}b"))))
                for i in range(1 << len(nums))]

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426184222.png)

### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：Itertools+压行，其实就是多个集合的笛卡尔积

代码：

```python
# coding: utf-8
"""
@File        :   phone_number_letter_17.py
@Time        :   2025/04/26 18:44:33
@Author      :   Usercyk
@Description :   Find all possible letter combinations that the number could represent.
"""
from itertools import product
from typing import List


class Solution:
    """
    The solution class
    """
    MAPPINGS = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    def letter_combinations(self, digits: str) -> List[str]:
        """
        All possible letter combinations that the number could represent.

        Arguments:
            digits -- 2-9 digit string

        Returns:
            List of all possible letter combinations
        """
        return ["".join(x) for x in product(
            *map(self.MAPPINGS.get, digits))] if digits else []  # type: ignore


print(Solution().letter_combinations("23"))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426185356.png)

### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：构建一个字典树，判断是否达到最后即可

代码：

```python
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

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426190538.png)

### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：建图的时候不能$O(N^2L)$，会超时

代码：

```python
# coding: utf-8
"""
@File        :   word_ladder_28046.py
@Time        :   2025/04/26 19:30:43
@Author      :   Usercyk
@Description :   Find the shortest transformation sequence from beginWord to endWord.
"""
from collections import defaultdict, deque
from typing import Dict, List, Optional


class Solution:
    """
    The solution class
    """

    def build_graph(self, word_list: List[str]) -> Dict[str, List[str]]:
        """
        Build a graph from the word list.

        Arguments:
            word_list -- the list of words

        Returns:
            A dictionary representing the graph
        """
        graph = defaultdict(list)
        pattern_map = defaultdict(list)
        for word in word_list:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                pattern_map[pattern].append(word)
        for word in word_list:
            neighbors = set()
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                neighbors.update(pattern_map[pattern])
            neighbors.discard(word)
            graph[word] = list(neighbors)
        return graph

    def find_shortest_transformation(self,
                                     begin_word: str,
                                     end_word: str,
                                     graph: Dict[str, List[str]]) -> Optional[List[str]]:
        """
        Find the shortest transformation sequence from begin_word to end_word.

        Arguments:
            begin_word -- The starting word
            end_word -- The ending word
            graph -- The graph built from the word list

        Returns:
            The path from begin_word to end_word, or an empty list if no path exists
        """
        queue = deque([(begin_word, [begin_word])])
        visited = {begin_word}
        while queue:
            current_word, path = queue.popleft()
            if current_word == end_word:
                return path
            for neighbor in graph[current_word]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def solve(self) -> None:
        """
        Solve the problem
        """
        word_list = [input().strip() for _ in range(int(input()))]
        begin_word, end_word = input().split()

        if begin_word not in word_list or end_word not in word_list:
            print("NO")
            return

        graph = self.build_graph(word_list + [begin_word, end_word])

        s = self.find_shortest_transformation(begin_word, end_word, graph)
        if s is None:
            print("NO")
        else:
            print(*s)


if __name__ == '__main__':
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426195533.png)

### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：正常写还是比较简单的，把之前的8皇后改成N就行了。于是我去学了一个位运算方法，用二进制数来替代list来标记。

代码：

```python
# coding: utf-8
"""
@File        :   n_queens_51.py
@Time        :   2025/04/26 19:07:05
@Author      :   Usercyk
@Description :   Solve N queens problem
"""

from typing import List


class Solution:
    """
    The solution class
    """

    def backtrack(self,
                  n: int,
                  row: int,
                  col: int,
                  left_diagonal: int,
                  right_diagonal: int,
                  board: List[str],
                  result: List[List[str]]
                  ) -> None:
        """
        Backtrack to find all solutions

        Arguments:
            n -- the number of queens
            row -- the current row
            col -- the current column
            left_diagonal -- the left diagonal
            right_diagonal -- the right diagonal
            board -- the current board
            result -- the list of solutions
        """
        if row == n:
            result.append(board[:])
            return

        for i in range(n):
            if all(((col & (1 << i)) == 0,
                   (left_diagonal & (1 << (row + i))) == 0,
                   (right_diagonal & (1 << (row - i + n - 1))) == 0)):
                board[row] = '.' * i + 'Q' + '.' * (n - i - 1)
                self.backtrack(n, row + 1, col | (1 << i), left_diagonal | (1 << (row + i)),
                               right_diagonal | (1 << (row - i + n - 1)), board, result)
                board[row] = '.' * n

    def solve_n_queens(self, n: int) -> List[List[str]]:
        """
        Solve N queens

        Arguments:
            n -- the number of queens

        Returns:
            A list of solutions, each solution is a list of strings
        """
        result = []
        board = ['.' * n for _ in range(n)]
        self.backtrack(n, 0, 0, 0, 0, board, result)
        return result

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250426191459.png)

## 2. 学习总结和收获

终于搞完期中考了，学习图ing
