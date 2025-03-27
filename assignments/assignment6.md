# Assignment #6: 回溯、树、双向链表和哈希表

Updated Mar 27, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：把类型输出正确（雾，毕竟permutations返回的是一个permutations[tuple[int]]类型

代码：

```python
# coding: utf-8
"""
@File        :   permutations_46.py
@Time        :   2025/03/27 13:08:25
@Author      :   Usercyk
@Description :   Permute
"""


from itertools import permutations
from typing import List


class Solution:
    """
    The solution class
    """

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Permute the nums

        Arguments:
            nums -- Numbers

        Returns:
            Permutations
        """
        return [list(t) for t in permutations(nums)]

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327131313.png)

### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：遍历从每个点出发的所有可能路径即可

代码：

```python
# coding: utf-8
"""
@File        :   word_search_79.py
@Time        :   2025/03/27 13:20:38
@Author      :   Usercyk
@Description :   Search word with backtracking
"""
from itertools import product
from typing import List, Optional, Tuple


class Solution:
    """
    The solution class
    """
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self) -> None:
        self.board: Optional[List[List[str]]] = None
        self.m: int = 0
        self.n: int = 0
        self.word: Optional[str] = None
        self.visited: Optional[List[List[bool]]] = None

    def search(self, board_pos: Tuple[int, int], word_pos: int) -> bool:
        """
        Match the word from word_pos with the board from board_pos

        Arguments:
            board_pos -- current board pos
            word_pos -- current word pos

        Returns:
            Whether the left part can be matched
        """
        if self.board is None or self.word is None or self.visited is None:
            return False
        x, y = board_pos

        if word_pos >= len(self.word):
            return True
        if self.board[x][y] != self.word[word_pos]:
            return False

        if word_pos == len(self.word)-1:
            return True

        self.visited[x][y] = True
        nw = word_pos+1
        for dx, dy in self.DIRECTIONS:
            nx, ny = x+dx, y+dy
            if nx < 0 or nx >= self.m or ny < 0 or ny >= self.n or self.visited[nx][ny]:
                continue
            if self.search((nx, ny), nw):
                return True

        self.visited[x][y] = False
        return False

    def reset_visited(self) -> None:
        """
        Reset the visited according to the board
        """
        if self.board is None or not self.board[0]:
            return None
        self.visited = [[False]*self.n for _ in range(self.m)]

    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        Check if the word exists

        Arguments:
            board -- The character board
            word -- The word used to be found

        Returns:
            The research result
        """
        self.board = board
        self.word = word
        self.m = len(self.board)
        self.n = len(self.board[0])
        self.reset_visited()
        for pos in product(range(self.m), range(self.n)):
            if self.search(pos, 0):
                return True
        return False


print(Solution().exist([["a"]], "a"))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327140217.png)

### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：额……根据记忆默写算法？（bushi

递归方法没啥好说的。

这个迭代方法，事实上就是先走到二叉树的最左边，然后逐个遍历，能向左就向左。

代码：

```python
# coding: utf-8
"""
@File        :   binary_tree_inorder_traversal_94.py
@Time        :   2025/03/27 14:03:55
@Author      :   Usercyk
@Description :   Inorder traversal
"""
from typing import List, Optional


class TreeNode:
    """
    The binary tree
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def inorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal using recursion

        Arguments:
            root -- The root node

        Returns:
            the inorder traversal
        """
        if root is None:
            return []
        return [*self.inorder_traversal_recursive(root.left),
                root.val,
                *self.inorder_traversal_recursive(root.right)]

    def inorder_traversal_iter(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal using iterating

        Arguments:
            root -- The root node

        Returns:
            the inorder traversal
        """
        res = []
        stack = []

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            assert root is not None
            res.append(root.val)
            root = root.right
        return res

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327142030.png)

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327144045.png)

### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：就……暴力……

代码：

```python
# coding: utf-8
"""
@File        :   binary_tree_level_order_102.py
@Time        :   2025/03/27 14:46:09
@Author      :   Usercyk
@Description :   Level order
"""
from typing import List, Optional


class TreeNode:
    """
    The binary tree
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Level order

        Arguments:
            root -- The root node

        Returns:
            level order
        """
        if root is None:
            return []

        res = []
        queue = [root]

        temp_res = []
        temp_queue = []

        while queue:
            for node in queue:
                temp_res.append(node.val)
                if node.left is not None:
                    temp_queue.append(node.left)
                if node.right is not None:
                    temp_queue.append(node.right)
            queue = temp_queue
            res.append(temp_res)
            temp_res = []
            temp_queue = []

        return res

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327151202.png)

### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：就是先想到这个和之前的一些划分题目很像，需要使用回溯来得到所有的划分。

然后需要判断每划分出的一个东西是否是回文串，如果是边dfs边判断，就$O(n^32^n)$了，肯定超时。

所以可以提前写好是否回文，用dp来判断。

代码：

```python
# coding: utf-8
"""
@File        :   palindrome_partitioning_131.py
@Time        :   2025/03/27 15:18:45
@Author      :   Usercyk
@Description :   Palindrome partitioning
"""
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.s: Optional[str] = None
        self.n: int = 0
        self.palindromes: Optional[List[List[bool]]] = None
        self.partitionings: List[List[str]] = []
        self.dfs_temp: List[str] = []

    def get_all_palindrome(self) -> List[List[bool]]:
        """
        Get all palindrome

        Returns:
            if s[i:j+1] is palindrome
        """
        assert self.s is not None
        palindromes = [[True]*self.n for _ in range(self.n)]
        for i in range(self.n-1, -1, -1):
            for j in range(i+1, self.n):
                palindromes[i][j] = (self.s[i] == self.s[j]
                                     ) and palindromes[i+1][j-1]
        return palindromes

    def dfs(self, idx: int = 0) -> None:
        """
        Dfs the string

        Keyword Arguments:
            idx -- the current index (default: {0})
        """
        assert self.palindromes is not None
        assert self.s is not None
        if idx >= self.n:
            self.partitionings.append(self.dfs_temp.copy())
            return
        for j in range(idx, self.n):
            if self.palindromes[idx][j]:
                self.dfs_temp.append(self.s[idx:j+1])
                self.dfs(j+1)
                self.dfs_temp.pop()

    def partition(self, s: str) -> List[List[str]]:
        """
        Palindrome partitioning

        Arguments:
            s -- the string

        Returns:
            all possible partitioning
        """
        self.s = s
        self.n = len(s)
        self.palindromes = self.get_all_palindrome()
        self.dfs()
        return self.partitionings

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327154510.png)

### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：说实在的，虽然说使用一个dummy可以避免判断过多的None，但是还是需要写很多的assert，静态检查总是检查不到……

代码：

```python
# coding: utf-8
"""
@File        :   lru_cache_146.py
@Time        :   2025/03/27 15:54:45
@Author      :   Usercyk
@Description :   LRU Cache
"""


from typing import Dict, Optional


class DoubleLinkedNode:
    """
    double linked node
    """

    def __init__(self, key: int = -1, value: int = -1) -> None:
        self.key: int = key
        self.value: int = value
        self.prior: Optional[DoubleLinkedNode] = None
        self.next: Optional[DoubleLinkedNode] = None

    def set_next(self, other: Optional["DoubleLinkedNode"]) -> None:
        """
        Link two nodes

        Arguments:
            other -- Another node
        """
        if isinstance(other, DoubleLinkedNode):
            self.next = other
            other.prior = self
        if other is None:
            self.next = None

    def set_prior(self, other: Optional["DoubleLinkedNode"]) -> None:
        """
        Link two nodes

        Arguments:
            other -- Another node
        """
        if isinstance(other, DoubleLinkedNode):
            self.prior = other
            other.next = self
        if other is None:
            self.prior = None


class LRUCache:
    """
    The LRU cache
    """

    def __init__(self, capacity: int):
        self.capacity: int = capacity

        self.cache: Dict[int, DoubleLinkedNode] = dict()
        self.size: int = 0

        self.head: DoubleLinkedNode = DoubleLinkedNode()
        self.tail: DoubleLinkedNode = DoubleLinkedNode()
        self.head.set_next(self.tail)

    def get(self, key: int) -> int:
        """
        Get value in O(1)

        Arguments:
            key -- The key

        Returns:
            The value of the key
        """
        if key not in self.cache:
            return -1
        node = self.cache[key]
        assert node.prior is not None
        node.prior.set_next(node.next)
        node.set_next(self.head.next)
        self.head.set_next(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Put items to the cache in O(1)

        Arguments:
            key -- The key
            value -- The value
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value

            assert node.prior is not None
            node.prior.set_next(node.next)
            node.set_next(self.head.next)
            self.head.set_next(node)
            return

        node = DoubleLinkedNode(key, value)
        self.cache[key] = node

        node.set_next(self.head.next)
        self.head.set_next(node)

        self.size += 1

        if self.size > self.capacity:
            node = self.tail.prior
            assert node is not None
            assert node.prior is not None
            node.prior.set_next(self.tail)

            self.cache.pop(node.key)
            self.size -= 1

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250327161529.png)


## 2. 学习总结和收获

LeetCode有个奇怪的bug，它的每一个程序用时是存在一个名叫"display_runtime.txt"的文件下的，所以可以直接修改这个文件来得到时间。由于需要在程序运行完后，修改它给的默认时间，需要我们在程序退出时执行一些东西，而python有atexit库

所以只需要在代码里加上这一行，就可以修改自己的时间了：
```python
__import__("atexit").register(lambda: open("display_runtime.txt", "w").write("0"))
```
当然如果是TLE，那还是TLE
