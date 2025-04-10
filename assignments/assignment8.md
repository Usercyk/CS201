# Assignment #8: 树为主

Updated  Apr 10, 2025

2025 spring, Complied by 曹以楷 物理学院


## 1. 题目

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：根节点取最中间的那个即可

代码：

```python
# coding: utf-8
"""
@File        :   covert_to_bst_108.py
@Time        :   2025/04/10 13:40:35
@Author      :   Usercyk
@Description :   Convert sorted array to binary search tree.
"""
from typing import List, Optional


class TreeNode:
    """
    The tree node
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val: int = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def sorted_array_to_bst(self, nums: List[int]) -> Optional[TreeNode]:
        """
        Convert sorted array to binary search tree.

        Arguments:
            nums -- The sorted array

        Returns:
            The binary search tree
        """
        if not nums:
            return None

        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sorted_array_to_bst(nums[:mid])
        root.right = self.sorted_array_to_bst(nums[mid + 1:])

        return root

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410134639.png)

### M27928:遍历树

adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：这个树的输入方式好麻烦……anyway，其实树的子节点采用两个优先队列就可以了。

代码：

```python
# coding: utf-8
"""
@File        :   traverse_27928.py
@Time        :   2025/04/10 13:56:58
@Author      :   Usercyk
@Description :   Traverse the tree in the "ascending" order.
"""
from heapq import heappop, heappush
from typing import Optional


class TreeNode:
    """
    Tree node
    """

    def __init__(self, val: int):
        self.val = val
        self.left_children = []
        self.right_children = []

    def add_child(self, child: 'TreeNode'):
        """
        Add a child to the tree node

        Arguments:
            child -- The child node to be added
        """
        if child < self:
            heappush(self.left_children, child)
        else:
            heappush(self.right_children, child)

    def __lt__(self, other: 'TreeNode') -> bool:
        """
        Compare two tree nodes

        Arguments:
            other -- The other tree node to be compared

        Returns:
            True if the current node is less than the other node, False otherwise
        """
        return self.val < other.val


class Solution:
    """
    The solution class
    """

    def __init__(self):
        self.nodes = {}

    def get_node(self, val: int, as_child: bool = False) -> TreeNode:
        """
        Get the node with the given value

        Arguments:
            val -- The value of the node

        Returns:
            The node with the given value
        """
        if val not in self.nodes:
            self.nodes[val] = (TreeNode(val), as_child)
        node, is_child = self.nodes[val]
        self.nodes[val] = (node, is_child or as_child)
        return node

    def build_tree(self) -> Optional[TreeNode]:
        """
        Build the tree

        Returns:
            The root of the tree
        """
        n = int(input())
        for _ in range(n):
            *a, = map(int, input().split())
            p = self.get_node(a.pop(0))
            for i in a:
                c = self.get_node(i, True)
                p.add_child(c)

        root = None

        for node, is_child in self.nodes.values():
            if not is_child:
                root = node
                break

        return root

    def traverse(self, root: Optional[TreeNode]) -> None:
        """
        Traverse the tree in the "ascending" order
        """
        if root is None:
            return
        while root.left_children:
            self.traverse(heappop(root.left_children))
        print(root.val)
        while root.right_children:
            self.traverse(heappop(root.right_children))


if __name__ == '__main__':
    sol = Solution()
    sol.traverse(sol.build_tree())

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410143826.png)

### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：叶子节点就直接返回，不是叶子就dfs

代码：

```python
# coding: utf-8
"""
@File        :   sum_root_to_leaf_numbers_129.py
@Time        :   2025/04/10 15:14:38
@Author      :   Usercyk
@Description :   Calculate the sum of all root-to-leaf numbers in a binary tree.
"""
from typing import Optional


class TreeNode:
    """
    The tree node
    """

    def __init__(self,
                 val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val: int = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class Solution:
    """
    The solution class
    """

    def sum_numbers(self, root: Optional[TreeNode]) -> int:
        """
        Calculate the sum of all root-to-leaf numbers in a binary tree.

        Arguments:
            root -- The root node of the binary tree

        Returns:
            The sum of all root-to-leaf numbers
        """
        return self.dfs(root, 0)

    def dfs(self, root: Optional[TreeNode], current: int) -> int:
        """
        Depth-first search to calculate the sum of all root-to-leaf numbers.

        Arguments:
            root -- The current node
            current -- The current number formed by the path from the root to this node

        Returns:
            The sum of all root-to-leaf numbers from this node
        """
        if root is None:
            return 0

        current = current * 10 + root.val

        if root.left or root.right:
            return self.dfs(root.left, current) + self.dfs(root.right, current)

        return current

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410152930.png)

### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：没有算法，全是递归（雾

找到根节点—>找到左右子树的前中序序列->拼接后序序列

代码：

```python
# coding: utf-8
"""
@File        :   pre_middle_to_post_22518.py
@Time        :   2025/04/10 15:34:17
@Author      :   Usercyk
@Description :   Convert pre and middle order traversal of a binary tree to post-order traversal.
"""


class Solution:
    """
    The solution class
    """

    def post_order(self, pre_order: str, middle_order: str) -> str:
        """
        Convert pre-order and middle-order traversal of a binary tree to post-order traversal.

        Arguments:
            pre_order -- The pre-order traversal string.
            middle_order -- The middle-order traversal string.

        Returns:
            The post-order traversal string.
        """
        if not pre_order or not middle_order:
            return ""

        root = pre_order[0]

        root_index = middle_order.index(root)

        left_subtree_pre = pre_order[1:1 + root_index]
        left_subtree_middle = middle_order[:root_index]
        right_subtree_pre = pre_order[1 + root_index:]
        right_subtree_middle = middle_order[root_index + 1:]

        return self.post_order(left_subtree_pre, left_subtree_middle) + \
            self.post_order(right_subtree_pre, right_subtree_middle) + root

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                pre_order = input().strip()
                middle_order = input().strip()
                print(self.post_order(pre_order, middle_order))
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410153916.png)

### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：用栈建树🤔

代码：

```python
# coding: utf-8
"""
@File        :   bracket_to_pre_post_order_24729.py
@Time        :   2025/04/10 15:43:10
@Author      :   Usercyk
@Description :   Convert a bracket expression to pre-order and post-order traversals.
"""
from typing import Optional


class TreeNode:
    """
    Tree node
    """

    def __init__(self, val: str):
        self.val = val
        self.children = []

    def add_child(self, child: 'TreeNode'):
        """
        Add a child to the tree node

        Arguments:
            child -- The child node to be added
        """
        self.children.append(child)


class Solution:
    """
    The solution class
    """

    def build_tree(self, bracket_expression: str):
        """
        Build a tree from a bracket expression.

        Arguments:
            bracket_expression -- The bracket expression to be converted.

        Examples:
            A(B(E),C(F,G),D(H(I)))
        """
        root = None
        stack = []
        current = None
        for c in bracket_expression:
            if c == "(":
                if current is not None:
                    stack.append(current)
            elif c == ")":
                if stack:
                    current = stack.pop()
            elif c == ',':
                if stack:
                    current = stack[-1]
            elif c.isalpha():
                node = TreeNode(c)
                if root is None:
                    root = node
                if current is not None:
                    current.add_child(node)
                current = node
        return root

    def pre_order(self, root: Optional[TreeNode]) -> str:
        """
        Pre-order traversal of the tree.

        Arguments:
            root -- The root of the tree.

        Returns:
            The pre-order traversal of the tree.
        """
        if root is None:
            return ""
        res = root.val
        for child in root.children:
            res += self.pre_order(child)
        return res

    def post_order(self, root: Optional[TreeNode]) -> str:
        """
        Post-order traversal of the tree.

        Arguments:
            root -- The root of the tree.

        Returns:
            The post-order traversal of the tree.
        """
        if root is None:
            return ""
        res = ""
        for child in root.children:
            res += self.post_order(child)
        res += root.val
        return res

    def solve(self) -> None:
        """
        Solve the problem
        """
        expr = input().strip()
        root = self.build_tree(expr)
        print(self.pre_order(root))
        print(self.post_order(root))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410160900.png)

### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：这题不愧是困难……

首先，使用优先队列来存储相邻元素的和，这一点应该是不难想到的，毕竟每次寻找最小和的开销显然是不可接受的。

其次，我卡在如何判断数组非递减，每次判断的开销显然也太大了……看了题解后发现其实使用一个dec计数器标记逆序对数量就行了……好吧我没想到………╮(╯-╰)╭

再次，就是要处理好在每次操作后的优先队列如何改变，肯定不能在创建一个新的，那么就要对原有队列进行修改。

具体地，比如优先队列里给出了`nums[i]`和`nums[j]`的和最小，j不一定是i+1，因为中间的数据可能被已经合并了。同样地，假设合并后变为`..., nums[i], nums[k], ...`，k同样不一定是j+1。所以需要额外维护一个当前剩余的下标数列。

最后，这个额外的下标数列也可以替换为双向链表，毕竟只是需要有方法获取到前一个和后一个存在的元素。但是不想写链表的代码，虽然可以用`nxt[i]=j, pre[j]=i`来模拟双向链表，但这次也没用这个，而是使用了leetcode环境中提供的`sortedcontainers`库，使用`python`实现了优先队列和其中的二分方法之类的。不出所料的，时间很慢……………下次还是不用了……

代码：

```python
# coding: utf-8
"""
@File        :   minimum_pair_removal_to_sort_3510.py
@Time        :   2025/04/10 17:04:59
@Author      :   Usercyk
@Description :   Minimum Pair Removal to Sort
"""
from itertools import pairwise
from typing import List

from sortedcontainers import SortedList


class Solution:
    """
    The solutino class
    """

    def minimum_pair_removal(self, nums: List[int]) -> int:
        """
        Calculate the minimum pair removal to sort

        Arguments:
            nums -- The list of integers

        Returns:
            The minimum pair removal to sort
        """
        pq = SortedList()
        idxs = SortedList(range(len(nums)))
        dec = 0

        for i, (x, y) in enumerate(pairwise(nums)):
            if x > y:
                dec += 1
            pq.add((x+y, i))

        cnt = 0
        while dec:
            cnt += 1
            s, i = pq.pop(0)
            k = idxs.bisect_left(i)

            nxt_i: int = idxs[k+1]  # type: ignore
            dec -= nums[i] > nums[nxt_i]

            if k:
                pre_i: int = idxs[k-1]  # type: ignore
                dec = dec-(nums[pre_i] > nums[i])+(nums[pre_i] > s)

                pq.remove((nums[pre_i]+nums[i], pre_i))
                pq.add((s+nums[pre_i], pre_i))

            if k+2 < len(idxs):
                nxt_nxt_i: int = idxs[k+2]  # type: ignore
                dec = dec-(nums[nxt_i] > nums[nxt_nxt_i])+(s > nums[nxt_nxt_i])

                pq.remove((nums[nxt_i]+nums[nxt_nxt_i], nxt_i))
                pq.add((s+nums[nxt_nxt_i], i))

            nums[i] = s
            idxs.remove(nxt_i)

        return cnt

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250410174202.png)

## 2. 学习总结和收获

在期中考ing……/(ㄒoㄒ)/~~
