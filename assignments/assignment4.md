# Assignment #4: 位操作、栈、链表、堆和NN

Updated Mar 17, 2025

2025 spring, Complied by 曹以楷 物理学院


## 1. 题目

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/

<mark>请用位操作来实现，并且只使用常量额外空间。</mark>

代码：这应该是一个很经典的方法了吧，利用异或来进行奇偶校验，比如Hamming Code汉明码也是类似的原理。

Anyway，其实就是出现两次的异或为0，其它数与0异或不变，异或具有交换律与结合律。所以直接全体异或就可以了。

至于这个常量额外空间🤔得看python怎么实现reduce了，应该是只用了1个int来存结果，依次异或下去。

哦，一般来说reduce需要自己写函数，但是所有类似于`lambda x, y: x+y`这种东西其实都已经在operator库里面有了

```python
# coding: utf-8
"""
@File        :   single_number_136.py
@Time        :   2025/03/17 13:36:56
@Author      :   Usercyk
@Description :   Find the number only exists once
"""
from functools import reduce
from operator import xor
from typing import List


class Solution:
    "The solution"

    def single_number(self, nums: List[int]) -> int:
        """
        Exclusive all to find the single number

        Arguments:
            nums -- All numbers

        Returns:
            The single number
        """
        return reduce(xor, nums)

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317134454.png)

### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/

思路：多次游程编码的解码器，说起来如果真的是论文的话，重复性很低，采用游程编码压缩肯定会越来越长的。

代码：

```python
# coding: utf-8
"""
@File        :   chemistry_article_20140.py
@Time        :   2025/03/17 13:59:38
@Author      :   Usercyk
@Description :   Decode the multiple run length encoding
"""


class Solution:
    """
    The solution
    """

    def decode(self, code: str):
        """
        Decode the mutiple RLE
        """
        stack = []
        curr_num, curr_str = 0, ""

        for c in code:
            if c.isdigit():
                curr_num = curr_num*10+int(c)
            elif c == "[":
                stack.append((curr_num, curr_str))
                curr_num, curr_str = 0, ""
            elif c == "]":
                prev_num, prev_str = stack.pop()
                curr_str = prev_str+curr_str*curr_num
                curr_num = prev_num
            else:
                curr_str += c

        return curr_str


if __name__ == "__main__":
    print(Solution().decode(input()))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317141158.png)

### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/

思路：两个指针在链表上，走skipA+skipB+both=m+n-both之后，一定均来到交叉点。

故时间复杂度O(m+n-both)，空间复杂度O(2)

不过leetcode上我试了半天都没用，时间上还是排在后面，然后我仔细一看……同样代码的运行时间可以差40ms……？？

代码：

```python
# coding: utf-8
"""
@File        :   get_intersection_node_160.py
@Time        :   2025/03/17 14:15:41
@Author      :   Usercyk
@Description :   Get the intersection node of two LinkNode
"""


from typing import Optional


class ListNode:
    """
    The list node
    """

    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    The solution
    """

    def get_intersection_node(self, head_a: ListNode, head_b: ListNode) -> Optional[ListNode]:
        """
        Get the intersection node of two single-linked list

        Arguments:
            head_a -- one single-linked list
            head_b -- another single-linked list

        Returns:
            The intersection node
        """
        pointer_a, pointer_b = head_a, head_b

        while pointer_a != pointer_b:
            pointer_a = head_b if pointer_a is None else pointer_a.next
            pointer_b = head_a if pointer_b is None else pointer_b.next

        return pointer_a

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317142624.png)

### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/

思路：遍历的方法，相当于把每一个链表的链条依次的反过来。递归的方法是，后续反过来，head再接在后面。

代码：

```python
# coding: utf-8
"""
@File        :   reverse_list_206.py
@Time        :   2025/03/17 14:36:15
@Author      :   Usercyk
@Description :   Reverse list
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

    def reverse_list_iter(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverst the list using iteration

        Arguments:
            head -- list

        Returns:
            reversed list
        """
        p1, p2 = None, None
        while head is not None:
            p2 = head.next
            head.next = p1
            p1 = head
            head = p2

        return p1

    def reverse_list_recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverst the list recursivly

        Arguments:
            head -- list

        Returns:
            reversed list
        """
        if head is None or head.next is None:
            return head

        tail = self.reverse_list_recursive(head.next)

        head.next.next = head
        head.next = None

        return tail

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317144127.png)

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317144454.png)

### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/

思路：先排序，之后就变成nums2的前k个最大的数。感觉nums1的用处就是增加代码长度而不是算法难度，要先排序，还要按原来顺序输出。

不过在同样时间复杂度的情况，这段代码似乎还有优化空间🤔从leetcode上最快的那个代码来看，排序tuple可能比较慢，而用num1作为key来排序idx是一个比较好的选择。

代码：

```python
# coding: utf-8
"""
@File        :   find_max_sum_2478.py
@Time        :   2025/03/17 14:47:57
@Author      :   Usercyk
@Description :   Find the max sum of k value
"""
from heapq import heappush, heappushpop
from typing import List


class Solution:
    """
    The solution
    """

    def find_max_sum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        Find the max sum

        Arguments:
            nums1 -- The first array
            nums2 -- The second array
            k -- k

        Returns:
            The max k sum
        """
        st = sorted((val, nums2[idx], idx)
                    for idx, val in enumerate(nums1))

        heap = []
        len_heap = 0

        res = 0
        ans = [0]*len(nums1)

        for i, (n1, n2, idx) in enumerate(st):
            if i > 0 and n1 == st[i-1][0]:
                ans[idx] = ans[st[i-1][2]]  # type: ignore
            else:
                ans[idx] = res
            res += n2

            if len_heap < k:
                heappush(heap, n2)
                len_heap += 1
            else:
                res -= heappushpop(heap, n2)

        return ans

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317152158.png)

### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

![](https://raw.githubusercontent.com/Usercyk/images/main/20250317153940.png)


我这个属于是力大砖飞，强制拟合，大概率过拟合了，但是loss确实降低了（


## 2. 学习总结和收获

复习链表和堆。