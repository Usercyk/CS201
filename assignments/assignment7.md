# Assignment #7: 20250402 Mock Exam

Updated Apr 6, 2025

2025 spring, Complied by 曹以楷 物理学院

AC6

## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/

思路：第N次考约瑟夫问题……

代码：

```python
# coding: utf-8
"""
@File        :   joseph_05344.py
@Time        :   2025/04/06 15:44:39
@Author      :   Usercyk
@Description :   Josephus Problem
"""
from collections import deque
from typing import List


class Solution:
    """
    The solution class
    """

    def josephus(self, n: int, k: int) -> List[int]:
        """
        Josephus problem

        Arguments:
            n -- number of people in the circle
            k -- step count

        Returns:
            The order of people being eliminated
        """
        res = []
        people = deque(range(1, n + 1))
        while len(people) > 1:
            people.rotate(-(k - 1))
            res.append(people.popleft())

        return res


if __name__ == "__main__":
    N, K = map(int, input().split())
    result = Solution().josephus(N, K)
    print(*result)

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406154944.png)

### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/

思路：看到描述为最大最小值，就知道需要二分了。

注：OJ的bisect还是不支持key啊……

代码：

```python
# coding: utf-8
"""
@File        :   cut_wood_02744.py
@Time        :   2025/04/06 15:52:26
@Author      :   Usercyk
@Description :   Cut the wood
"""
from typing import List


class Solution:
    """
    The solution
    """

    def check(self, length: int, woods: List[int], k: int) -> bool:
        """
        Check if the wood can be cut

        Arguments:
            length -- The length of wood
            woods -- The length of woods
            k -- The number of pieces

        Returns:
            True if the wood can be cut, False otherwise
        """
        return sum(wood // length for wood in woods) >= k

    def cut_wood(self, k: int, woods: List[int]) -> int:
        """
        Cut the wood

        Arguments:
            k -- The number of pieces
            woods -- The length of woods

        Returns:
            The max length of wood
        """
        left, right = 1, max(woods)
        result = 0
        while left <= right:
            mid = (left + right) // 2
            if self.check(mid, woods, k):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result


if __name__ == "__main__":
    N, K = map(int, input().split())
    Woods = [int(input()) for _ in range(N)]
    print(Solution().cut_wood(K, Woods))

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406160009.png)

### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/

思路：就……实现呗，用个队列临时存存就行了

代码：

```python
# coding: utf-8
"""
@File        :   forest_sequence_with_degree_07161.py
@Time        :   2025/04/06 16:04:02
@Author      :   Usercyk
@Description :   Forest hierarchical sequence storage with degrees
"""

from collections import deque
from typing import List, Optional


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
    The solution
    """

    def build_tree(self, sequence: list) -> Optional[TreeNode]:
        """
        Build the tree from the sequence

        Arguments:
            sequence -- The sequence of the tree

        Returns:
            The root node of the tree

        Example:
            sequence: C 3 E 3 F 0 G 0 K 0 H 0 J 0
            results:
                C
                ├── E
                │   ├── K
                │   ├── H
                │   └── J
                ├── F
                ├── G
        """
        if not sequence:
            return None

        root = TreeNode(sequence[0])
        q = deque([(root, int(sequence[1]))])
        i = 2

        while i < len(sequence):
            node, degree = q.popleft()
            for _ in range(degree):
                child = TreeNode(sequence[i])
                node.add_child(child)
                q.append((child, int(sequence[i + 1])))
                i += 2

        return root

    def post_order_traversal(self, node: Optional[TreeNode]) -> List[str]:
        """
        Post-order traversal of the tree

        Arguments:
            node -- The root node of the tree

        Returns:
            The post-order traversal of the tree
        """
        if node is None:
            return []

        result = []
        for child in node.children:
            result.extend(self.post_order_traversal(child))

        result.append(node.val)
        return result

    def solve(self) -> None:
        """
        Solve the problem
        """
        n = int(input())
        for _ in range(n):
            sequence = input().split()
            root = self.build_tree(sequence)
            result = self.post_order_traversal(root)
            print(' '.join(result), end=' ')


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406162641.png)

### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/

思路：两个指针从外往里移动。这个“如果存在多个解，则输出数值较小的那个”…唉…多个判断

代码：

```python
# coding: utf-8
"""
@File        :   closest_sum_18156.py
@Time        :   2025/04/06 16:28:46
@Author      :   Usercyk
@Description :   Closest Sum
"""
from typing import List


class Solution:
    """
    The solution class
    """

    def closest_sum(self, nums: List[int], target: int) -> int:
        """
        Find the closest sum to the target.

        Arguments:
            nums -- list of integers
            target -- target integer

        Returns:
            The closest sum to the target.
        """
        nums.sort()
        left, right = 0, len(nums) - 1
        ans = float('inf')
        while left < right:
            current = nums[left] + nums[right]
            if abs(current - target) < abs(ans - target):
                ans = current
            elif abs(current - target) == abs(ans - target):
                ans = min(ans, current)

            if current < target:
                left += 1
            elif current > target:
                right -= 1
            else:
                break
        return int(ans)

    def solve(self) -> None:
        """
        Solve the problem
        """
        target = int(input())
        nums = list(map(int, input().split()))
        print(self.closest_sum(nums, target))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406163557.png)

### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/

思路：筛法，筛过一次后就可以打表了（雾

代码：

```python
# coding: utf-8
"""
@File        :   prime_ending_1_18159.py
@Time        :   2025/04/06 16:40:12
@Author      :   Usercyk
@Description :   Find the prime number that ends with 1.
"""
from typing import List


class Solution:
    """
    The solution class
    """
    PRIMES_END_WITH_1 = [11, 31, 41, 61, 71,...]

    def __init__(self) -> None:
        self.primes_end_with_1 = []
        # self.euler_sieve(10001)
        self.primes_end_with_1 = self.PRIMES_END_WITH_1

    def euler_sieve(self, n: int) -> None:
        """
        Find all prime numbers that end with 1 up to n.

        Arguments:
            n -- the upper limit
        """
        is_prime = [True] * (n + 1)
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                if i % 10 == 1:
                    self.primes_end_with_1.append(i)
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    break

    def find(self, n: int) -> List[int]:
        """
        Find all prime numbers that end with 1 up to n.

        Arguments:
            n -- The upper limit

        Returns:
            A list of prime numbers that end with 1 up to n.
        """
        res = []
        for i in self.primes_end_with_1:
            if i >= n:
                break
            res.append(i)
        return res

    def solve(self) -> None:
        """
        Solve the problem
        """
        t = int(input())
        for i in range(t):
            n = int(input())
            print(f"Case{i + 1}:")
            if n < 12:
                print("NULL")
            else:
                res = self.find(n)
                print(" ".join(map(str, res)))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406170328.png)

### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/

思路：这个题就很有面向对象的风格。创建一个对象ICPC，用于对University进行排名，对每个收到的提交，去通知对应University收到一个更新，最后将结果排序后输出。而University自己要做的就是，在收到更新后统计自己的结果。

这下是真·举办(hold)ICPC即可

代码：

```python
# coding: utf-8
"""
@File        :   pku_champion_28127.py
@Time        :   2025/04/06 17:06:25
@Author      :   Usercyk
@Description :   Sort the universities in ICPC
"""


class University:
    """
    The university class
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.questions = {k: False for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        self.update_times = 0

    def update(self, question: str, passed: str) -> None:
        """
        Update the university with the question and passed status

        Arguments:
            question -- The question name
            passed -- The passed status
        """
        self.questions[question] = self.questions[question] or passed == "yes"
        self.update_times += 1

    @property
    def score(self) -> int:
        """
        Calculate the score of the university

        Returns:
            The score of the university
        """
        return sum(1 for k in self.questions if self.questions[k])

    def __str__(self) -> str:
        return f"{self.name} {self.score} {self.update_times}"

    def __lt__(self, other: 'University') -> bool:
        """
        Compare two universities

        Arguments:
            other -- The other university

        Returns:
            True if self is less than other, False otherwise
        """
        if self.score != other.score:
            return self.score > other.score
        if self.update_times != other.update_times:
            return self.update_times < other.update_times
        return self.name < other.name


class ICPC:
    """
    The icpc competition
    """

    def __init__(self) -> None:
        self.universities = {}

    def update(self, submission: str) -> None:
        """
        Update the university with the submission

        Arguments:
            submission -- The submission string
        """
        name, question, passed = submission.split(",")
        if name not in self.universities:
            self.universities[name] = University(name)
        self.universities[name].update(question, passed)

    def hold(self) -> None:
        """
        Run the contest
        """
        for _ in range(int(input())):
            submission = input().strip()
            self.update(submission)

        for i, university in enumerate(sorted(self.universities.values())):
            if i >= 12:
                break
            print(f"{i + 1} {university}")


if __name__ == "__main__":
    ICPC().hold()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250406172343.png)

## 2. 学习总结和收获

这次考试给我的感觉像是复习？复习之前写的内容🤔算法好像不是很多，基本都是考实现。
