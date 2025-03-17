# Assignment #3: 惊蛰 Mock Exam

Updated Mar 7, 2025

2025 spring, Complied by 曹以楷 物理学院

AC 6

## 1. 题目

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015

思路：之前做这个题的时候是使用正则表达式的，但是由于是考试，正则表达式不太熟练导致错误时调试时间比较长，不如直接判断。

代码：

```python
# coding: utf-8
"""
@File        :   validate_email_04015.py
@Time        :   2025/03/07 13:09:45
@Author      :   Usercyk
@Description :   validate email
"""


class Solution:
    """
    The solution class
    """

    def validate(self, s: str) -> bool:
        """
        Validate email

        Arguments:
            s -- The email string

        Returns:
            True if the email is valid, False otherwise
        """
        split_email = s.strip().split('@')
        if len(split_email) != 2:
            return False
        head, tail = split_email
        if head == '' or tail == '':
            return False
        if head[0] == '.' or head[-1] == '.' or tail[0] == "." or tail[-1] == ".":
            return False
        return "." in tail

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                email = input()
                print("YES" if self.validate(email) else "NO")
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307134900.png)

### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/

思路：分行后直接输出，分段可以用自带库。建议不要学习我这种遍历过程中修改列表的坏习惯。

代码：

```python
# coding: utf-8
"""
@File        :   reverse_decode_02039.py
@Time        :   2025/03/07 14:08:51
@Author      :   Usercyk
@Description :   Decode
"""
import textwrap


class Solution:
    """
    The solution class
    """

    def solve(self, col: int, message: str):
        """
        Solve the problem
        """
        s = [list(x) for x in textwrap.wrap(message, col)]
        for _ in range(col):
            for idx, val in enumerate(s):
                print(val.pop(-(idx % 2)), end="")


if __name__ == "__main__":
    Solution().solve(int(input()), input())

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307141933.png)

### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/

思路：倒排索引

代码：

```python
# coding: utf-8
"""
@File        :   granpa_is_famous_02092.py
@Time        :   2025/03/07 14:23:07
@Author      :   Usercyk
@Description :   Get the 2nd best player
"""
from typing import List
from collections import defaultdict


class Solution:
    """
    The solution class
    """
    MAX_PLAYER = 10001

    def get_2nd_player(self, players: List[int]) -> List[int]:
        """
        Get the second players

        Arguments:
            players -- all ranked players

        Returns:
            The number of the second player
        """
        rank = defaultdict(int)
        for player in players:
            rank[player] += 1
        invert = defaultdict(list)
        for k, v in rank.items():
            invert[v].append(k)

        second_time = sorted(invert.keys())[-2]
        return sorted(invert[second_time])

    def solve(self):
        """
        Solve the problem
        """
        while True:
            n, m = map(int, input().split())
            if n == m == 0:
                break
            players = []
            for _ in range(n):
                players.extend(list(map(int, input().split())))

            ans = self.get_2nd_player(players)
            print(*ans, end=" \n")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307185710.png)

### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/

思路：由于并不涉及多枚炸弹，仅一枚的情况下直接~~暴力~~遍历即可

代码：

```python
# coding: utf-8
"""
@File        :   rubbish_bomb_04133.py
@Time        :   2025/03/07 13:55:11
@Author      :   Usercyk
@Description :   Rubbish bomb
"""


from itertools import product
from typing import List, Tuple


class Solution:
    """
    The solution class
    """

    def explode(self, d: int, rubbishes: List[Tuple[int, int, int]], bomb_cord: Tuple[int, int]) -> int:
        """
        Explode the bomb

        Arguments:
            d -- The power of the bomb
            rubbishes -- The rubbish in the street. (x, y, rubbish_cnt)
            bomb_cord -- The place where the bomb explode. (bomb_x, bomb_y)

        Returns:
            The count of cleared rubbish
        """
        bomb_x, bomb_y = bomb_cord
        res = 0
        for rubbish_x, rubbish_y, rubbish_cnt in rubbishes:
            if bomb_x-d <= rubbish_x <= bomb_x+d and bomb_y-d <= rubbish_y <= bomb_y+d:
                res += rubbish_cnt
        return res

    def solve(self):
        """
        solve the problem
        """
        d = int(input())
        rubbishes = []
        for _ in range(int(input())):
            rubbishes.append(tuple(map(int, input().split())))
        ans = -1
        cnt = 0
        for p in product(range(1025), range(1025)):
            res = self.explode(d, rubbishes, p)
            if res > ans:
                ans = res
                cnt = 1
            elif res == ans:
                cnt += 1
        print(cnt, ans)


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307140647.png)

### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/

思路：回溯即可，重点是需要排列好自己的方向使得第一个弄出来的就是字典序第一个

代码：

```python
# coding: utf-8
"""
@File        :   knights_journey_02488.py
@Time        :   2025/03/07 18:59:38
@Author      :   Usercyk
@Description :   Get the possible paths for a knight to pass every squares in pxq board.
"""


class Solution:
    """
    The solution class
    """
    MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    def __init__(self) -> None:
        self.path = []
        self.p = -1
        self.q = -1
        self.board = []
        self.flag = False

    def travel(self, step: int = 1, x: int = 0, y: int = 0) -> bool:
        """
        Travel the pxq board

        Arguments:
            step -- the current step
            x -- current pos x
            y -- current pos y

        Returns:
            Can the knight travel through all the board
        """
        if step == self.p*self.q:
            self.flag = True
            return True

        for dy, dx in self.MOVES:
            nx, ny = x+dx, y+dy

            if all((not self.flag, 0 <= nx < self.p, 0 <= ny < self.q)):
                if self.board[nx][ny] != 1:
                    self.board[nx][ny] = 1
                    self.path[step] = (nx, ny)
                    self.travel(step+1, nx, ny)
                    self.board[nx][ny] = 0

        return self.flag

    def re_init(self, p: int, q: int):
        """
        Init the board and paths

        Arguments:
            p -- the numbers
            q -- the alphabets
        """
        self.p, self.q = p, q
        self.path = [(0, 0) for _ in range(p*q)]

        self.board = [[0]*(q+1) for _ in range(p+1)]
        self.board[0][0] = 1

        self.flag = False

    def solve(self):
        """
        Solve the problem
        """
        for i in range(int(input())):
            self.re_init(*map(int, input().split()))

            print(f"Scenario #{i+1}:")
            if self.travel():
                ans = (chr(c[1]+ord("A"))+str(c[0]+1) for c in self.path)
                print("".join(ans))
            else:
                print("impossible")
            print("")


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307194932.png)

### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/

思路：使用优先队列（最小堆）来保存前n个最小值，然后归并

代码：

```python
# coding: utf-8
"""
@File        :   sequence_06648.py
@Time        :   2025/03/07 19:56:35
@Author      :   Usercyk
@Description :   Merge Sequence
"""
import heapq
from typing import List


class Solution:
    """
    The solution class
    """

    def __init__(self):
        self.n = 0

    def merge(self, a: List[int], b: List[int]) -> List[int]:
        """
        Merge the sorted array to get first n minimal sum

        Arguments:
            a -- a sorted array
            b -- another sorted array

        Returns:
            first n minimal sum
        """
        heap = []
        visited = set()

        heapq.heappush(heap, (a[0]+b[0], 0, 0))
        visited.add((0, 0))

        res = []
        len_a = len(a)
        len_b = len(b)

        while heap:
            if len(res) == self.n:
                break
            s, i, j = heapq.heappop(heap)
            res.append(s)
            if i+1 < len_a and (i+1, j) not in visited:
                heapq.heappush(heap, (a[i+1]+b[j], i+1, j))
                visited.add((i+1, j))
            if j+1 < len_b and (i, j+1) not in visited:
                heapq.heappush(heap, (a[i]+b[j+1], i, j+1))
                visited.add((i, j+1))

        return res

    def solve(self):
        """
        Solve the problem
        """
        for _ in range(int(input())):
            m, self.n = map(int, input().split())
            arr = [sorted(map(int, input().split())) for _ in range(m)]
            ans = arr.pop()
            while arr:
                ans = self.merge(ans, arr.pop())
            print(*ans)


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250307201133.png)

## 2. 学习总结和收获

感觉难度比计概难（这大概是废话）

月考因为调鸢尾花调着调着，一抬头就错过了…
星期五才找到时间做，但因为有课导致不是连续的两个小时🤔
所以断断续续地也算是卡点完成了月考吧。
主要调Knight调得时间比较长，花了差不多四五十分钟吧。主要一开始p,q反了所以字典序不对了。
