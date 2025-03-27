# Assignment #5: 链表、栈、队列和归并排序

Updated Mar 24, 2025

2025 spring, Complied by 曹以楷 物理学院
## 1. 题目

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：递归即可，最多O(m+n)

代码：

```python
# coding: utf-8
"""
@File        :   merge_two_lists_21.py
@Time        :   2025/03/23 23:21:10
@Author      :   Usercyk
@Description :   Merge two sorted lists
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

    def merge_two_lists(self,
                        list1: Optional[ListNode],
                        list2: Optional[ListNode]
                        ) -> Optional[ListNode]:
        """
        Merge two sorted lists

        Arguments:
            list1 -- list 1
            list2 -- list 2

        Returns:
            merged list
        """
        if list1 is None:
            return list2
        if list2 is None:
            return list1

        if list1.val < list2.val:
            list1.next = self.merge_two_lists(list1.next, list2)
            return list1

        list2.next = self.merge_two_lists(list1, list2.next)
        return list2

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250323232511.png)

### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

思路：感觉时间复杂度是O(1.5N)，空间复杂度是O(4)，所以应该还有优化空间……？

算法就是，先到一半，然后后面反过来，再判断回文。

这里发现由于if只判断了快指针，所以慢的几个的类型检查都有问题，可以assert一下。

代码：

```python
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

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250323233646.png)

### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>

思路：双链表实现是实现了……但似乎远远不如直接使用一个list来存，每次判断应该去到的下标快

我为了不每次设定一遍next和prior，写了一个setter，但似乎导致创建对象的开销有点大了……

代码：

```python
# coding: utf-8
"""
@File        :   design_browser_history_1472.py
@Time        :   2025/03/23 23:40:36
@Author      :   Usercyk
@Description :   Design a browser history
"""
from typing import Optional


class DoubleListNode:
    """
    Double list node
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._prior: Optional[DoubleListNode] = None
        self._next: Optional[DoubleListNode] = None

    @property
    def prior(self) -> Optional["DoubleListNode"]:
        """
        _prior getter

        Returns:
            self._prior
        """
        return self._prior

    @property
    def next(self) -> Optional["DoubleListNode"]:
        """
        _next getter

        Returns:
            self._next
        """
        return self._next

    @next.setter
    def next(self, other: Optional["DoubleListNode"]) -> None:
        self._next = other
        if other is not None:
            other.set_prior(self)

    @prior.setter
    def prior(self, other: Optional["DoubleListNode"]) -> None:
        self._prior = other
        if other is not None:
            other.set_next(self)

    def set_next(self, other: Optional["DoubleListNode"]) -> None:
        """
        Set next avoid infinite recursion

        Arguments:
            other -- the next node
        """
        self._next = other

    def set_prior(self, other: Optional["DoubleListNode"]) -> None:
        """
        Set prior avoid infinite recursion

        Arguments:
            other -- the prior node
        """
        self._prior = other


class BrowserHistory:
    """
    The browser history
    """

    def __init__(self, homepage: str):
        """
        Initialize the browser history

        Arguments:
            homepage -- the homepage
        """
        self.current = DoubleListNode(homepage)

    def visit(self, url: str) -> None:
        """
        Visit the url, clear the forward history

        Arguments:
            url -- the url
        """
        v = DoubleListNode(url)
        self.current.next = v
        self.current = v

    def back(self, steps: int) -> str:
        """
        Go back steps in history, return the url.

        Arguments:
            steps -- Up to steps back

        Returns:
            the url
        """
        while self.current.prior is not None and steps:
            self.current = self.current.prior
            steps -= 1

        return self.current.url

    def forward(self, steps: int) -> str:
        """
        Go forward steps in history, return the url.

        Arguments:
            steps -- Up to steps forward

        Returns:
            the url
        """
        while self.current.next is not None and steps:
            self.current = self.current.next
            steps -= 1

        return self.current.url

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250324000402.png)

### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：就还是基础的入栈出栈

代码：

```python
# coding: utf-8
"""
@File        :   infix_to_postfiix_24591.py
@Time        :   2025/03/24 00:40:00
@Author      :   Usercyk
@Description :   Convert infix to postfix
"""
from typing import List


class Solution:
    """
    The solution class
    """

    PRIORITY = {"+": 1, "-": 1, "*": 2, "/": 2}
    NUMBERS = "0123456789."

    def infix_to_postfix(self, infix: str) -> List[str]:
        """
        Conver the infix to postfix

        Arguments:
            infix -- The infix

        Returns:
            The converted postfix
        """
        i = 0
        len_infix = len(infix)
        stack = []
        res = []

        while i < len_infix:
            if infix[i] in self.NUMBERS:
                j = i
                while j < len_infix and infix[j] in self.NUMBERS:
                    j += 1
                num = infix[i:j]
                res.append(num)
                i = j
            elif infix[i] == "(":
                stack.append("(")
                i += 1
            elif infix[i] == ")":
                while stack and stack[-1] != "(":
                    res.append(stack.pop())
                stack.pop()
                i += 1
            else:
                oper = infix[i]
                while stack and stack[-1] != '(' and \
                        self.PRIORITY[oper] <= self.PRIORITY.get(stack[-1], 0):
                    res.append(stack.pop())
                stack.append(oper)
                i += 1
        while stack:
            res.append(stack.pop())
        return res

    def solve(self):
        """
        Solve the problem
        """
        for _ in range(int(input())):
            print(*self.infix_to_postfix(input()))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250324005409.png)

### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>

思路：双端队列算不算队列（

代码：

```python
# coding: utf-8
"""
@File        :   josephus_03253.py
@Time        :   2025/03/24 00:15:39
@Author      :   Usercyk
@Description :   Solve josephus problem
"""
from collections import deque
from typing import List


class Solution:
    """
    The solution class
    """

    def josephus(self, n: int, p: int, m: int) -> List[int]:
        """
        Solve josephus problem

        Arguments:
            n -- n kids
            p -- p start
            m -- m out

        Returns:
            The order of getting out
        """
        queue = deque(range(1, n+1))
        queue.rotate(1-p)

        result = []

        while queue:
            for _ in range(m-1):
                queue.append(queue.popleft())
            result.append(queue.popleft())

        return result

    def solve(self):
        """
        Solve the problem
        """
        while True:
            n, p, m = map(int, input().split())
            if n == p == m == 0:
                break
            print(*self.josephus(n, p, m), sep=",")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250324002603.png)

### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：经典的归并排序求逆序对（或顺序对）

代码：

```python
# coding: utf-8
"""
@File        :   ant_run_20018.py
@Time        :   2025/03/24 13:35:34
@Author      :   Usercyk
@Description :   Calculate the count of ordered pair
"""
from typing import List


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.count = 0

    def merge_count(self, sequence: List[int]) -> List[int]:
        """
        Count the ordered pair and sort the sequence

        Arguments:
            sequence -- original sequence
        """
        if len(sequence) <= 1:
            return sequence

        mid = len(sequence)//2
        left_seq = self.merge_count(sequence[:mid])
        right_seq = self.merge_count(sequence[mid:])

        sorted_seq = []
        i, j = 0, 0

        left_len = len(left_seq)
        right_len = len(right_seq)

        while i < left_len and j < right_len:
            if left_seq[i] < right_seq[j]:
                self.count += left_len-i
                sorted_seq.append(right_seq[j])
                j += 1
            else:
                sorted_seq.append(left_seq[i])
                i += 1

        sorted_seq.extend(left_seq[i:])
        sorted_seq.extend(right_seq[j:])

        return sorted_seq

    def solve(self) -> None:
        """
        Solve the problem
        """
        seq = [int(input()) for _ in range(int(input()))]
        self.count = 0
        self.merge_count(seq)
        print(self.count)
        self.count = 0


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250324135008.png)

## 2. 学习总结和收获

考试的时候会给链表题吗🤔感觉OJ上没有提供默认的ListNode？
