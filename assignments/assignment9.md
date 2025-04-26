# Assignment #9: Huffman, BST & Heap

Updated Apr 21, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：直接递归比较简单，题解给出来的这个位运算的方法挺好

代码：

```python
# coding: utf-8
"""
@File        :   count_complete_btree_222.py
@Time        :   2025/04/21 19:16:52
@Author      :   Usercyk
@Description :   Complete Binary Tree Node Count
"""
from typing import Optional


class TreeNode:
    """
    Binary tree
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    The solution class
    """

    def find(self, root: Optional[TreeNode], level: int, tag: int) -> bool:
        """
        Find if the corresponding node exists in the complete binary tree.

        Arguments:
            root -- The root of the binary tree
            level -- The level of the binary tree
            tag -- The tag of the binary tree

        Returns:
            True if the node exists, False otherwise
        """
        bits = 1 << (level-1)
        node = root
        while node is not None and bits > 0:
            if bits & tag:
                node = node.right
            else:
                node = node.left
            bits >>= 1

        return node is not None

    def count_nodes(self, root: Optional[TreeNode]) -> int:
        """
        Count the number of nodes in a complete binary tree.

        Arguments:
            root -- The root of the binary tree

        Returns:
            The number of nodes in the complete binary tree
        """
        if root is None:
            return 0

        h = 0
        node = root
        while node is not None:
            h += 1
            node = node.left

        l = 1 << (h-1)
        r = (1 << h)-1
        while l < r:
            mid = (l+r+1)//2
            if self.find(root, h-1, mid):
                l = mid
            else:
                r = mid-1

        return l

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250421194933.png)

### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：假的锯齿，其实多个reverse就行

代码：

```python
# coding: utf-8
"""
@File        :   zigzag_order_103.py
@Time        :   2025/04/21 19:52:12
@Author      :   Usercyk
@Description :   Zigzag Level Order Traversal of a Binary Tree
"""
from typing import List, Optional


class TreeNode:
    """
    Binary tree
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    The solution class
    """

    def zigzag_level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Zigzag level order traversal of a binary tree.

        Arguments:
            root -- The root of the binary tree

        Returns:
            A list of lists of integers representing the zigzag level order traversal
        """
        if not root:
            return []

        ans = []
        queue = [root]
        left_to_right = True

        while queue:
            nq = []
            vals = []
            for node in queue:
                vals.append(node.val)
                if node.left is not None:
                    nq.append(node.left)
                if node.right is not None:
                    nq.append(node.right)
            if not left_to_right:
                vals.reverse()
            ans.append(vals)
            queue = nq
            left_to_right = not left_to_right
        return ans

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250421195854.png)

### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：实现

代码：

```python
# coding: utf-8
"""
@File        :   huffman_tree_04080.py
@Time        :   2025/04/21 20:10:34
@Author      :   Usercyk
@Description :   Build a Huffman tree from a list of weights
"""
import heapq
from typing import List


class Solution:
    """
    The solution class
    """

    def min_weighted_path_length(self, weights: List[int]) -> int:
        """
        Calculate the minimum weighted path length of a Huffman tree.

        Arguments:
            weights -- a list of weights

        Returns:
            The minimum weighted path length of the Huffman tree
        """
        heap = weights.copy()
        heapq.heapify(heap)
        ans = 0

        while len(heap) > 1:
            w1, w2 = heapq.heappop(heap), heapq.heappop(heap)
            ans += w1+w2
            heapq.heappush(heap, w1+w2)

        return ans


if __name__ == "__main__":
    input()
    print(Solution().min_weighted_path_length(list(map(int, input().split()))))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250421201437.png)

### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：实现BST即可

代码：

```python
# coding: utf-8
"""
@File        :   level_order_of_bst_05455.py
@Time        :   2025/04/21 20:22:10
@Author      :   Usercyk
@Description :   Build a binary search tree and print its level order traversal.
"""
from typing import List, Optional


class TreeNode:
    """
    Binary tree
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    The solution class
    """

    def level_order(self, root: Optional[TreeNode]) -> List[int]:
        """
        Perform a level order traversal of the binary search tree.

        Arguments:
            root -- the root of the tree

        Returns:
            A list of values in level order
        """
        if not root:
            return []

        queue = [root]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def build(self, array: List[int]) -> Optional[TreeNode]:
        """
        Build a binary search tree from the given array.

        Arguments:
            array -- the array to build the tree from

        Returns:
            The root of the binary search tree
        """
        if not array:
            return None

        root = TreeNode(array[0])
        for i in range(1, len(array)):
            self.insert(root, array[i])
        return root

    def insert(self, root: TreeNode, val: int) -> None:
        """
        Insert a value into the binary search tree.

        Arguments:
            root -- the root of the tree
            val -- the value to insert
        """
        if val == root.val:
            return
        if val < root.val:
            if root.left is None:
                root.left = TreeNode(val)
            else:
                self.insert(root.left, val)
        else:
            if root.right is None:
                root.right = TreeNode(val)
            else:
                self.insert(root.right, val)


if __name__ == "__main__":
    sol = Solution()
    rt = sol.build(list(map(int, input().split())))
    res = sol.level_order(rt)
    print(*res)

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250421212751.png)

### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：其实可以查看heapq的源代码来着🤔不过堆也很经典了，这里给加了一个key。

代码：

```python
# coding: utf-8
"""
@File        :   heap_04078.py
@Time        :   2025/04/22 12:42:24
@Author      :   Usercyk
@Description :   Heap
"""
from typing import List


class Heap:
    """
    A simple implementation of a heap.
    """

    def __init__(self, key=None) -> None:
        self._array: List[int] = []
        self._key = key if key is not None else lambda x: x
        self._size = 0

    def left(self, i: int) -> int:
        """
        Returns the index of the left child of the node at index i.
        """
        return 2 * i + 1

    def right(self, i: int) -> int:
        """
        Returns the index of the right child of the node at index i.
        """
        return 2 * i + 2

    def parent(self, i: int) -> int:
        """
        Returns the index of the parent of the node at index i.
        """
        if i == 0:
            raise IndexError("Root has no parent")
        return (i - 1) // 2

    def insert(self, value: int) -> None:
        """
        Inserts a new value into the heap.
        """
        self._array.append(value)
        self._size += 1
        self._sift_up(self._size - 1)

    def heappop(self) -> int:
        """
        Removes and returns the smallest value from the heap.
        """
        if self._size == 0:
            raise IndexError("Heap is empty")
        root = self._array[0]
        last_element = self._array.pop()
        self._size -= 1
        if self._size > 0:
            self._array[0] = last_element
            self._sift_down(0)
        return root

    def _cmp(self, i: int, j: int) -> bool:
        """
        Compares the values at indices i and j in the heap.
        """
        return self._key(self._array[i]) < self._key(self._array[j])

    def _sift_up(self, i: int) -> None:
        """
        Moves the node at index i up to its correct position in the heap.
        """
        while i > 0 and self._cmp(i, self.parent(i)):
            self._array[i], self._array[self.parent(
                i)] = self._array[self.parent(i)], self._array[i]
            i = self.parent(i)

    def _sift_down(self, i: int) -> None:
        """
        Moves the node at index i down to its correct position in the heap.
        """
        while True:
            left = self.left(i)
            right = self.right(i)
            smallest = i

            if left < self._size and self._cmp(left, smallest):
                smallest = left
            if right < self._size and self._cmp(right, smallest):
                smallest = right
            if smallest == i:
                break
            self._array[i], self._array[smallest] = self._array[smallest], self._array[i]
            i = smallest

    def __len__(self) -> int:
        """
        Returns the number of elements in the heap.
        """
        return self._size

    def to_list(self) -> List[int]:
        """
        Returns the elements of the heap as a list.
        """
        return self._array[:self._size]

    def __str__(self) -> str:
        """
        Returns a string representation of the heap.
        """
        return str(self.to_list())


class Manager:
    """
    A simple manager for the heap.
    """

    def __init__(self, key=None) -> None:
        self._heap = Heap(key)

    def work(self, op: str) -> None:
        """
        Deal with the heap operations based on the type and value provided.

        Arguments:
            op -- typ val
            typ -- 1: insert, 2: pop
            val -- the value to be inserted or ignored if type is 2
        """
        typ = int(op[0])
        if typ == 1:
            val = int(op[2:])
            self._heap.insert(val)
        elif typ == 2:
            if len(self._heap) > 0:
                print(self._heap.heappop())
            else:
                raise IndexError("Heap is empty")
        else:
            raise ValueError("Invalid operation type")


if __name__ == "__main__":
    manager = Manager()
    for _ in range(int(input())):
        manager.work(input().strip())

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250422130713.png)

### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：实现。写代码的时候出现的一个问题就是我试图使用set来存储节点的字符集，但是需要一个获取节点中字符的方式，使用set的pop方法会损坏这个set，最后使用了next(iter(set))的方式。

代码：

```python
# coding: utf-8
"""
@File        :   huffman_code_tree_22161.py
@Time        :   2025/04/22 13:52:31
@Author      :   Usercyk
@Description :   Realize a Huffman code tree
"""
from heapq import heapify, heappop, heappush
from typing import List, Optional, Set


class Node:
    """
    Node in the Huffman tree.
    """

    def __init__(self, chars: Set[str], freq: int) -> None:
        self.chars = chars
        self.freq = freq
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def __lt__(self, other: 'Node') -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        return min(self.chars) < min(other.chars)


class HuffmanTree:
    """
    Huffman tree.
    """

    def __init__(self, chars: List[str], freqs: List[int]) -> None:
        self.chars = chars
        self.freqs = freqs
        self.root = None
        self.code_map = {}
        self.build_tree()
        self.build_code_map(self.root, "")

    def build_tree(self) -> None:
        """
        Build the Huffman tree from the characters and their frequencies.
        """
        nodes = [Node({char}, freq)
                 for char, freq in zip(self.chars, self.freqs)]
        heapify(nodes)
        while len(nodes) > 1:
            nodes.sort()
            left = heappop(nodes)
            right = heappop(nodes)
            merged = Node(left.chars.union(right.chars),
                          left.freq + right.freq)
            merged.left = left
            merged.right = right
            heappush(nodes, merged)
        self.root = nodes[0]

    def build_code_map(self, node: Optional[Node], path: str) -> None:
        """
        Build the code map for the characters in the Huffman tree.

        Arguments:
            node -- node to traverse
            path -- the path to the current node
        """
        if node is None:
            return
        if not node.left and not node.right:  # Leaf node
            self.code_map[next(iter(node.chars))] = path
            return
        self.build_code_map(node.left, path + "0")
        self.build_code_map(node.right, path + "1")

    def encode(self, string: str) -> str:
        """
        Encode the string using the Huffman tree.

        Arguments:
            string -- the string to encode

        Returns:
            The encoded string.
        """
        encoded_string = "".join(self.code_map[char] for char in string)
        return encoded_string

    def decode(self, string: str) -> str:
        """
        Decode the string using the Huffman tree.

        Arguments:
            string -- the string to decode

        Returns:
            The decoded string.
        """
        decoded_string = ""
        node = self.root
        for bit in string:
            if node is None:
                raise ValueError("Invalid encoded string")
            if bit == "0":
                node = node.left
            else:
                node = node.right
            if node is None:
                raise ValueError("Invalid encoded string")
            if not node.left and not node.right:
                decoded_string += next(iter(node.chars))
                node = self.root
        return decoded_string

    @classmethod
    def from_input_string(cls) -> 'HuffmanTree':
        """
        Create a Huffman tree from the input string.
        """
        chars = []
        freqs = []
        for _ in range(int(input())):
            c, f = input().split()
            f = int(f)
            chars.append(c)
            freqs.append(f)
        return cls(chars, freqs)


if __name__ == "__main__":
    huffman = HuffmanTree.from_input_string()
    while True:
        try:
            s = input()
            if "1" in s or "0" in s:
                print(huffman.decode(s))
            else:
                print(huffman.encode(s))
        except EOFError:
            break

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250422141253.png)

## 2. 学习总结和收获

期中考怎么还没考完/(ㄒoㄒ)/~~
