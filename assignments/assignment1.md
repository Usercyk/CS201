# Assignment #1: 虚拟机，Shell & 大语言模型

Updated Feb 23, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/

思路：有理数在定义上就是整数对$\mathbb{Z}\times \mathbb{Z}$的划分，所以只需要维护两个`int`即可

代码：

```python
# coding: utf-8
"""
@File        :   fraction_27653.py
@Time        :   2025/02/23 13:53:51
@Author      :   Usercyk
@Description :   realize the fraction class
"""


class Fraction:
    """
    Fraction class
    """
    @staticmethod
    def gcd(num_1: int, num_2: int) -> int:
        """
        Calculate the greatest common divisor of a and b

        Arguments:
            num_1 -- One of the number
            num_2 -- Another number

        Returns:
            The greatest common divisor of a and b
        """
        if num_2 == 0:
            return num_1
        return Fraction.gcd(num_2, num_1 % num_2)

    def _simplify(self) -> None:
        """
        Simplify the fraction
        """
        g = self.gcd(self.numerator, self.denominator)
        self.numerator //= g
        self.denominator //= g
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def __init__(self, numerator: int, denominator: int) -> None:
        """
        Initialize the fraction

        Arguments:
            numerator -- The numerator
            denominator -- The denominator
        """
        self.numerator = numerator
        if denominator == 0:
            raise ValueError("Denominator can't be zero")
        self.denominator = denominator
        self._simplify()

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __add__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator + other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __sub__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator - other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __mul__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.numerator,
                        self.denominator * other.denominator)

    def __truediv__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator,
                        self.denominator * other.numerator)


if __name__ == "__main__":
    a, b, c, d = map(int, input().split())
    print(Fraction(a, b) + Fraction(c, d))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250223142006.png)

### 1760.袋子里最少数目的球

https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/

思路：既然是要求最小最大值，那肯定二分即可

代码：

```python
# coding: utf-8
"""
@File        :   minimum_size_1760.py
@Time        :   2025/02/23 17:45:19
@Author      :   Usercyk
@Description :   Bisect to calculate the minimum size
"""
from bisect import bisect
from typing import Callable, List


class Solution:
    """
    The solution
    """

    def bisect_key(self, nums: List[int], max_operations: int) -> Callable[[int], int]:
        """
        Generate the bisect key from the numbers

        Arguments:
            nums -- the array
            max_operations -- the possible operation times

        Returns:
            The key function for bisect
        """
        return lambda x: max_operations-sum((num-1)//x for num in nums)+1

    def minimum_size(self, nums: List[int], max_operations: int) -> int:
        """
        Calculate the minimum size of the largest value in the array after operation

        Arguments:
            nums -- the array
            max_operations -- the possible operation times

        Returns:
            The minimum size of the maximum value after operation
        """
        return bisect(range(1, max(nums)+1), 0, key=self.bisect_key(nums, max_operations))+1

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250223183439.png)

### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135

思路：又是最小最大值？？那继续二分吧，说起来OJ python3.8的bisect没有key……

代码：

```python
# coding: utf-8
"""
@File        :   fajo_04135.py
@Time        :   2025/02/23 18:38:42
@Author      :   Usercyk
@Description :   Minimum value of the maximum cost of fajo months
"""
from functools import reduce
from typing import Callable, List


class Solution:
    """
    The solution
    """

    @staticmethod
    def bisect(a, x, key):
        """
        Copy of the 3.13.2 bisect.bisect
        """
        lo = 0
        hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def bisect_key(self, m: int, costs: List[int]) -> Callable[[int], int]:
        """
        Generate the bisect key from the costs

        Arguments:
            m -- The maximum number of the fajo months
            costs -- The cost of every day

        Returns:
            The key function for bisect
        """
        def func(previous, cur):
            pre_fajo, pre_cost, x = previous
            if cur+pre_cost > x:
                return (pre_fajo+1, cur, x)
            return (pre_fajo, pre_cost+cur, x)

        return lambda x: m-reduce(func, costs, (0, 0, x))[0]

    def solve(self, m: int, costs: List[int]) -> int:
        """
        Solve the minimum value of the maximum cost of fajo months

        Arguments:
            m -- The maximum number of the fajo months
            costs -- The cost of every day

        Returns:
            The minimum value of the maximum cost of fajo months
        """
        ran = range(max(costs), sum(costs)+1)
        return ran[self.bisect(ran, 0, key=self.bisect_key(m, costs))]


if __name__ == "__main__":
    N, M = map(int, input().split())
    Costs = [int(input()) for _ in range(N)]
    print(Solution().solve(M, Costs))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250223190540.png)

### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/

思路：这个题之前做过吧，贴一个之前的代码

代码：

```python
from collections import defaultdict

nums = {"M": 1_000_000, "B": 1_000_000_000}


def transform(s):
    return float(s[:-1]) * nums[s[-1]]


d = defaultdict(list)
for _ in range(int(input())):
    name, cnt = input().split("-")
    d[name].append(cnt)
for k in sorted(d.keys()):
    d[k].sort(key=transform)
    print(f"{k}: {', '.join(d[k])}")

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250223190731.png)

### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。





### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。





## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>





