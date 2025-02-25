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

1. 创建好虚拟机后，使用vscode进行连接
![](https://raw.githubusercontent.com/Usercyk/images/main/20250225090543.png)
2. 由于Rocky Linux缺少一些自带库，对其进行安装
```bash
sudo yum upgrade
sudo yum install wget -y
sudo yum install gcc gcc-c++ -y
sudo yum install git -y
```
3. 下载miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
4. 配置环境
```bash
conda create -n coder python=3.12
conda activate coder
```
5. 连接ITS
```python
#!/usr/bin/env python3
"""
@File        :   login.py
@Time        :   2025/02/25 08:07:47
@Author      :   CLab
@Description :   Login the ITS to connect the internet
"""
import requests
import getpass

# 从命令行获取用户名和密码
username = input("请输入用户名: ")
password = getpass.getpass("请输入密码: ")

url = "https://its4.pku.edu.cn/cas/ITSClient"
payload = {
    'username': username,
    'password': password,
    'iprange': 'free',
    'cmd': 'open'
}
headers = {'Content-type': 'application/x-www-form-urlencoded'}

result = requests.post(url, params=payload, headers=headers)
print(result.text)
```
6. 下载vscode插件
![](https://raw.githubusercontent.com/Usercyk/images/main/20250225091136.png)
7. 下载一些库
```bash
pip install --upgrade pip # 更新pip
pip install modelscope # 模型下载
pip install autopep8 # 自动整理
```
8. 下载模型DeepSeek-Coder-V2-Lite-Instruct
```python
#!/usr/bin/env python3
"""
@File        :   download.py
@Time        :   2025/02/24 18:56:49
@Author      :   Usercyk
@Description :   download the model
"""
from modelscope import snapshot_download

if __name__ == "__main__":
    model_dir = snapshot_download(
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        cache_dir='model/')

```
9. 由于虚拟机上只有虚拟显卡，提供一些基础的GUI渲染什么的，没有真正的GPU，所以只能采用cpu运行大模型。
10. 下载vllm
```bash
git clone https://github.com/vllm-project/vllm.git
```
11. 构建vllm的cpu后端
```bash
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python setup.py install
```
12. 运行最后一句会占用大量CPU，以至于ssh都没有足够cpu运行而卡顿。在重启云虚拟机后，发现搭建进程完成了一半且有缓存，尝试继续运行，然后就被kill了……甚至连vllm的后端都搭不起来……下次试试不用vllm

#### 问题
1. sda4爆满：迁移到数据盘并创建符号链接
```bash
sudo mv /home/rocky/model /mnt/data/model
sudo mv /home/rocky/vllm /mnt/data/vllm

sudo ln -s /mnt/data/vllm /home/rocky/vllm
sudo ln -s /mnt/data/model /home/rocky/model
```


### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

没来得及看，一直在调云虚拟机并破防


## 2. 学习总结和个人收获

正在完成寒假期间的每日选做
