# Assignment #2: 深度学习与大语言模型

Updated Mar 1, 2025

2025 spring, Complied by 曹以楷 物理学院

## 1. 题目

### 18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161

思路：这也是一道之前做过的题啊

代码：

```python
def getMatrices():
    matrices = [[], [], []]
    for i in range(3):
        row, _ = map(int, input().split())
        for _ in range(row):
            matrices[i].append(list(map(int, input().split())))
    return matrices


def times(a, b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b)))) for j in range(len(b[0]))] for i in range(len(a))]


def plus(c, d):
    return [[c[i][j]+d[i][j] for j in range(len(c[0]))] for i in range(len(c))]


def main():
    a, b, c = getMatrices()
    if not (len(a[0]) == len(b) and len(a) == len(c) and len(b[0]) == len(c[0])):
        print("Error!")
        return
    s = plus(c, times(a, b))
    for p in s:
        print(*p)


main()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250301160659.png)

### 19942: 二维矩阵上的卷积运算

matrices, http://cs101.openjudge.cn/practice/19942/

思路：这还是一道之前做过的题诶

代码：

```python
m, n, p, q = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(m)]
b = [list(map(int, input().split())) for _ in range(p)]

for i in range(m+1-p):
    resi = []
    for j in range(n+1-q):
        res = 0
        for k in range(p):
            for l in range(q):
                res += a[i+k][j+l]*b[k][l]
        resi.append(res)
    print(*resi)

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250301160808.png)

### 04140: 方程求解

牛顿迭代法，http://cs101.openjudge.cn/practice/04140/

思路：先算一下迭代的方程$$\phi(x)=x-\frac{x^3-5x^2+10x-80}{3x^2-10x+10}=\frac{2x^3-5x^2+80}{3x^2-10x+10}$$
与此同时，利用一点点数学，可以发现原方程有且仅有一实根，且在5到6之间。

代码：

```python
# coding: utf-8
"""
@File        :   newton_04140.py
@Time        :   2025/03/01 16:14:57
@Author      :   Usercyk
@Description :   Using Newton's method to solve the equation f(x) = 0
"""


class Solution:
    """
    The solution
    """

    def phi(self, x: float) -> float:
        """
        The iteration function

        Arguments:
            x -- The nth x value

        Returns:
            The (n+1)th x value
        """
        return (2*x**3-5*x**2+80)/(3*x**2-10*x+10)

    def solve(self, x_init: float = 5.0, eps: float = 1e-15) -> float:
        """
        Solve the equation f(x) = x**3-5*x**2+10*x-80 = 0

        Keyword Arguments:
            x_init -- The initial value of x (default: {5.0})
            eps -- The precision (default: {1e-15})

        Returns:
            The solution of the equation
        """
        x = x_init
        while True:
            x_next = self.phi(x)
            if abs(x_next-x) < eps:
                return x_next
            x = x_next


if __name__ == "__main__":
    print(f"{Solution().solve():.9f}")

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250301162502.png)

### 06640: 倒排索引

data structures, http://cs101.openjudge.cn/practice/06640/

思路：python的dict挺快的，直接用dict就行了。不过就算再慢也不会超过O(n)，全部被查询时的排序覆盖了。另一方面，利用set自动去重。

代码：

```python
# coding: utf-8
"""
@File        :   invert_index_06640.py
@Time        :   2025/03/01 16:56:07
@Author      :   Usercyk
@Description :   Realize the inverted index of the document
"""
from typing import Dict, List, Optional, Set


class Solution:
    """
    The Solution
    """

    def __init__(self) -> None:
        self.inverted_index: Dict[str, Set[int]] = {}

    def update(self, document_index: int, document: str) -> None:
        """
        Update the inverted index with the given document

        Arguments:
            document_index -- the index of the document
            document -- the content of the document
        """
        keywords = document.split()[1:]
        for keyword in keywords:
            cur: Set[int] = self.inverted_index.get(keyword, set())
            cur.add(document_index)
            self.inverted_index[keyword] = cur

    def query(self, keyword: str) -> Optional[List[int]]:
        """
        Query the index of the documents that contain the keyword

        Arguments:
            keyword -- The keyword

        Returns:
            The sorted index of the documents that contain the keyword
        """
        idx = self.inverted_index.get(keyword, None)
        if idx is not None:
            return sorted(idx)
        return None

    def solve(self) -> None:
        """
        Solve the problem
        """
        t = int(input())

        for i in range(t):
            self.update(i+1, input())
        q = int(input())

        for _ in range(q):
            res = self.query(input())
            if res is None:
                print("NOT FOUND")
            else:
                print(*res)


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250301171156.png)

### 04093: 倒排索引查询

data structures, http://cs101.openjudge.cn/practice/04093/

思路：集合的运算，唯一一点就是在于集合的初值得是全集，这个太麻烦了，于是使用一个数组存储没来得及作差集的那些集合。

代码：

```python
# coding: utf-8
"""
@File        :   query_inverted_index_04093.py
@Time        :   2025/03/01 17:21:07
@Author      :   Usercyk
@Description :   Query the inverted index of the document
"""

from typing import List, Optional, Set


class Solution:
    """
    The Solution
    """

    def __init__(self) -> None:
        self.inverted_index: List[Set[int]] = []

    def update(self, idxs: str) -> None:
        """
        Update the inverted index with the given indices

        Arguments:
            idxs -- The indices
        """
        self.inverted_index.append(set(map(int, idxs.split()[1:])))

    def query(self, requirement: str) -> Optional[Set[int]]:
        """
        Query the indices of the documents that fit the requirement

        Arguments:
            requirement -- The requirement

        Returns:
            The indices of the documents that fit the requirement
        """
        req = map(int, requirement.split())
        wait = []
        res: Optional[Set[int]] = None

        for i, r in enumerate(req):
            if r == 0:
                continue
            if r == 1:
                if res is None:
                    res = self.inverted_index[i]
                else:
                    res = res.intersection(self.inverted_index[i])
            elif r == -1:
                if res is None:
                    wait.append(i)
                else:
                    res = res.difference(self.inverted_index[i])

        for w in wait:
            if res is not None:
                res = res.difference(self.inverted_index[w])

        return res

    def solve(self):
        """
        Solve the problem
        """
        t = int(input())
        for _ in range(t):
            self.update(input())
        q = int(input())
        for _ in range(q):
            res = self.query(input())
            if res is None or len(res) == 0:
                print("NOT FOUND")
            else:
                print(*sorted(res))


if __name__ == "__main__":
    Solution().solve()

```

![](https://raw.githubusercontent.com/Usercyk/images/main/20250301173440.png)

### Q6. Neural Network实现鸢尾花卉数据分类

鸢尾花啊，一道非常经典的机器学习的入门题目。由于其数据集仅有150条，来自于一位植物学家Fisher，故我们的模型架构可以简单很多。除输入输出外，中间仅有一层。

#### 模型建立并训练

##### Iris的模型

```python
class IrisModel(nn.Module):
    """
    The class to define the model for iris dataset
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model

        Arguments:
            x -- The tensor input

        Returns:
            The tensor output
        """
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)

```

##### 训练超参数

```python
class TrainConfig:
    """
    The configuration class for training the model
    """
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    NUM_FOLDS: int = 5
    REG_LAMBDA: float = 0.01
    HIDDEN_SIZE: int = 10

```

##### Trainer

1. 定义标准化类和交叉熵损失
```python
self.scaler = StandardScaler()
self.criterion = nn.CrossEntropyLoss()
```
2. 引入k折验证判断模型的泛化性
```python
kfold = StratifiedKFold(
    n_splits=TrainConfig.NUM_FOLDS,
    shuffle=True,
    random_state=seed
)
```
3. 引入L2正则化，防止过拟合。同时由于数据量较小，采取SGD优化。
```python
optimizer = optim.SGD(
                model.parameters(),
                lr=TrainConfig.LEARNING_RATE,
                weight_decay=TrainConfig.REG_LAMBDA
            )
```
4. 对不同的随机种子进行多次实验以判断模型效果
```python
def train(self):
    """
    Train the model

    Returns:
        The average accuracy of the model
    """
    accuracies = []

    for seed in tqdm([42, 43, 44], desc="Seeds"):
        accuracies.extend(self.train_with_seed(seed))

    final_acc = np.mean(accuracies)
    print("\n=== Final Metrics ===")
    print(f"Average Accuracy: {final_acc:.4f}")
    print(f"Std Deviation: {np.std(accuracies):.4f}")
    print(f"Individual Accuracies: {[round(acc,4) for acc in accuracies]}")

    return final_acc
```
5. Trainer整体代码
```python
class Trainer:
    """
    Trainer class to train the model
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.criterion = nn.CrossEntropyLoss()

    def train_with_seed(self, seed) -> List[float]:
        """
        Train the model with certain seed

        Arguments:
            seed -- The random seed

        Returns:
            The accuracies of each fold
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        x, y = load_iris(return_X_y=True)
        x = self.scaler.fit_transform(x)  # type: ignore
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        kfold = StratifiedKFold(
            n_splits=TrainConfig.NUM_FOLDS,
            shuffle=True,
            random_state=seed
        )

        accuracies = []
        progress_bar = tqdm(
            kfold.split(x, y),
            total=TrainConfig.NUM_FOLDS,
            desc="CV Progress"
        )

        for fold, (train_idx, val_idx) in enumerate(progress_bar):
            x_train, x_val = x_tensor[train_idx], x_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            train_loader = DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=TrainConfig.BATCH_SIZE,
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=TrainConfig.BATCH_SIZE
            )

            model = IrisModel(
                input_size=x.shape[1],
                hidden_size=TrainConfig.HIDDEN_SIZE,
                output_size=len(np.unique(y))
            )
            optimizer = optim.SGD(
                model.parameters(),
                lr=TrainConfig.LEARNING_RATE,
                weight_decay=TrainConfig.REG_LAMBDA
            )

            epoch_bar = trange(
                TrainConfig.NUM_EPOCHS,
                desc=f"Fold {fold+1}",
                leave=False
            )
            for _ in epoch_bar:
                model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            fold_acc = correct / total
            accuracies.append(fold_acc)
            progress_bar.set_postfix_str(f"Acc: {fold_acc:.4f}")

        return accuracies

    def train(self):
        """
        Train the model

        Returns:
            The average accuracy of the model
        """
        accuracies = []

        for seed in tqdm([42, 43, 44], desc="Seeds"):
            accuracies.extend(self.train_with_seed(seed))

        final_acc = np.mean(accuracies)
        print("\n=== Final Metrics ===")
        print(f"Average Accuracy: {final_acc:.4f}")
        print(f"Std Deviation: {np.std(accuracies):.4f}")
        print(f"Individual Accuracies: {[round(acc,4) for acc in accuracies]}")

        return final_acc
```

##### Logger

为了将结果保存在文件中，定义Logger辅助输出。且tqdm的进度条默认输出至标准错误stderr，所以我们重定向标准输出stdout即可。

```python
class Logger:
    """
    The class to log the output both to terminal and the file
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1, encoding='utf-8')

    def write(self, message):
        """
        Write the message to both terminal and the file

        Arguments:
            message -- The message to write
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flush the output
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """
        Close the file
        """
        self.log.close()

```

##### 调整超参数

在完成上述步骤后，已经可以初步运行，但无法判断不同超参数对模型效果的影响。我们可以遍历所有超参数的取法，每一个都训练一遍，随后将结果保存。

```python
class HyperparameterTuner:
    """
    The class to tune the hyperparameters of the model
    """

    def __init__(self, param_grid: Dict[str, list]):
        self.param_grid = param_grid
        self.logger = DynamicLogger()
        self.results = []

    def generate_combinations(self):
        """
        Generate all possible combinations of hyperparameters

        Yields:
            The next combination of hyperparameters
        """
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _print_config_header(self, params):
        """
        Print the header of the current hyperparameters

        Arguments:
            params -- The current hyperparameters
        """
        print("=" * 50)
        print("Current Hyperparameters:")
        for k, v in params.items():
            print(f"{k}: {v}")
        print("=" * 50 + "\n")

    def run(self):
        """
        Run the hyperparameter tuning process
        """
        original_config = {k: getattr(TrainConfig, k)
                           for k in self.param_grid.keys()}

        try:
            for params in self.generate_combinations():

                filename = self._generate_filename(params)

                with self.logger.log_to_file(filename):
                    self._set_config(params)
                    self._print_config_header(params)

                    trainer = Trainer()
                    final_accuracy = trainer.train()

                    self.results.append({
                        **params,
                        'accuracy': final_accuracy,
                        'log_file': filename
                    })
        finally:
            self._restore_config(original_config)

        self._print_final_summary()

    def _set_config(self, params):
        for key, value in params.items():
            setattr(TrainConfig, key, value)

    def _restore_config(self, original):
        for key, value in original.items():
            setattr(TrainConfig, key, value)

    def _generate_filename(self, params):
        parts = []
        for k, v in params.items():
            if isinstance(v, float):
                v_str = f"{v:.0e}".replace('.', '').replace('e-0', 'e-')
            else:
                v_str = str(v)
            parts.append(f"{k}_{v_str}")
        return f"/home/rocky/assignment2/log/iris_{'_'.join(parts)}.log"

    def _print_final_summary(self):
        print("\n\n=== Hyperparameter Tuning Summary ===")
        for result in self.results:
            print(f"\nConfiguration: {result['log_file']}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("-" * 50)

```

##### DynamicLogger

为了配合超参数调整后，每次都需要重定向至不同的文件，再次引入一个辅助输出类。

```python
class DynamicLogger:
    """
    The class to log the output to dynamic files
    """

    def __init__(self):
        self.original_stdout = sys.stdout
        self.current_log = None

    @contextmanager
    def log_to_file(self, filename):
        """
        Log the output to a certain file

        Arguments:
            filename -- The filename to log the output
        """
        try:
            self.current_log = Logger(filename)
            sys.stdout = self.current_log
            yield
        finally:
            self.current_log.close()  # type: ignore
            sys.stdout = self.original_stdout

```

##### 调用并运行

```python
if __name__ == "__main__":
    parameters = {
        'LEARNING_RATE': [0.0001, 0.001, 0.01],
        'HIDDEN_SIZE': [5, 10, 20],
        'REG_LAMBDA': [0.001, 0.01, 0.1],
        'NUM_EPOCHS': [50, 100, 200]
    }

    tuner = HyperparameterTuner(parameters)
    tuner.run()
```

#### 结果呈现

##### 指定种子——42
```
Top 5 Best Result:
 LEARNING_RATE  HIDDEN_SIZE  REG_LAMBDA  NUM_EPOCHS  accuracy
          0.01           20       0.010         200    0.9200
          0.01           20       0.001         200    0.9200
          0.01           10       0.001         200    0.9133
          0.01           10       0.010         200    0.9067
          0.01            5       0.010         200    0.8800

Best parameters:
LEARNING_RATE: 0.01
HIDDEN_SIZE: 20.0
REG_LAMBDA: 0.01
NUM_EPOCHS: 200.0
Ave accuracy: 0.9200
```

##### 3个种子——42，43，44
```
Top 5 Best Result:
 LEARNING_RATE  HIDDEN_SIZE  REG_LAMBDA  NUM_EPOCHS  accuracy
          0.01           20       0.010         200    0.9044
          0.01           20       0.001         200    0.9022
          0.01           10       0.001         200    0.9000
          0.01           10       0.010         200    0.8956
          0.01           20       0.001         100    0.8800

Best parameters:
LEARNING_RATE: 0.01
HIDDEN_SIZE: 20.0
REG_LAMBDA: 0.01
NUM_EPOCHS: 200.0
Ave accuracy: 0.9044
```

可以看到，在不同种子的情况下，该超参数的训练结果仍然最好，且精确度在不同种子情况下与第二名产生了差距，说明L2正则化确实使其泛化能力增强了。

##### 可视化代码

写一段小小的代码将结果可视化。

```python
# coding: utf-8
"""
@File        :   show_res.py
@Time        :   2025/03/05 19:20:55
@Author      :   Usercyk
@Description :   Visualize the results of iris_tuner.py
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

LOG_DIR = "assignment2/log"
OUTPUT_IMG = "assignment2/hyperparam_analysis.png"


def parse_log_filename(filename):
    """
    Using the filename to parse the hyperparameters

    Arguments:
        filename -- The filename of the log file

    Returns:
        The dictionary of the hyperparameters
    """
    params = {}
    parts = filename.replace(".log", "").split("_")

    current_key = None
    for part in parts[1:]:
        if part in ["LEARNING", "HIDDEN", "REG", "NUM"]:
            current_key = part
        elif part in ["RATE", "SIZE", "LAMBDA", "EPOCHS"]:
            current_key += "_" + part  # type: ignore
        elif current_key:
            if 'e-' in part:
                params[current_key] = float(part.replace('e-', 'e-'))
            elif part.isdigit():
                params[current_key] = int(part)
            else:
                try:
                    params[current_key] = float(part)
                except Exception:  # pylint: disable=broad-except
                    params[current_key] = part
            current_key = None
    return params


def extract_accuracy(log_path):
    """
    Extract the accuracy from the log file

    Arguments:
        log_path -- The path of the log file

    Returns:
        The accuracy of the model
    """
    with open(log_path, 'r', encoding="utf-8") as f:
        content = f.read()
        match = re.search(r"Average Accuracy: (\d+\.\d+)", content)
        if match:
            return float(match.group(1))
    return None


def load_all_results():
    """
    Load all the results from the log files

    Returns:
        The DataFrame of the results
    """
    results = []

    for filename in os.listdir(LOG_DIR):
        if filename.endswith(".log"):
            filepath = os.path.join(LOG_DIR, filename)
            params = parse_log_filename(filename)
            accuracy = extract_accuracy(filepath)

            if accuracy is not None:
                results.append({
                    **params,
                    "accuracy": accuracy
                })

    return pd.DataFrame(results)


def plot_hyperparam_analysis(df):
    """
    Plot the hyperparameter sensitivity analysis

    Arguments:
        df -- The DataFrame of the results
    """
    plt.figure(figsize=(20, 16))
    plt.suptitle("Hyperparameter Sensitivity Analysis", fontsize=18)

    subplot_config = [
        ('LEARNING_RATE', 'log', 'Learning Rate'),
        ('REG_LAMBDA', 'log', 'L2 Regularization'),
        ('HIDDEN_SIZE', 'linear', 'Hidden Layer Size'),
        ('NUM_EPOCHS', 'linear', 'Training Epochs')
    ]

    for idx, (param, scale, title) in enumerate(subplot_config, 1):
        ax = plt.subplot(2, 2, idx)

        other_params = [p for p in df.columns if p not in [param, 'accuracy']]
        grouped = df.groupby(other_params)

        for name, group in grouped:
            sorted_group = group.sort_values(param)
            ax.plot(sorted_group[param], sorted_group['accuracy'],
                    marker='o', linestyle='--', alpha=0.7,
                    label=f"{', '.join([f'{k}={v}' for k,v in zip(other_params, name)])}")

        ax.set_xscale(scale)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"The result is saved to: {OUTPUT_IMG}")


def find_best_parameters(df):
    """
    Find the best hyperparameters

    Arguments:
        df -- The DataFrame of the results
    """
    best_idx = df['accuracy'].idxmax()
    best_params = df.loc[best_idx].to_dict()

    print("\nBest parameters:")
    for k, v in best_params.items():
        if k == 'accuracy':
            print(f"Ave accuracy: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    data_frame = load_all_results()

    data_frame = data_frame.sort_values('accuracy', ascending=False)

    print("Top 5 Best Result:")
    print(data_frame.head(5).to_string(index=False))

    find_best_parameters(data_frame)

    plot_hyperparam_analysis(data_frame)

```

##### 可视化结果（三个种子）

![](https://raw.githubusercontent.com/Usercyk/images/main/hyperparam_analysis.png)

##### 结论
1. 学习率越高越准。
2. 隐藏层的大小越大，最后的结果一般会越好。
3. L2正则化系数越高，其结果越差，但泛化性越好，防止过拟合。
4. 训练轮次越多，拟合得越好。


## 2. 学习总结和个人收获

每日选做补不完了，在做了在做了/(ㄒoㄒ)/~~

感觉鸢尾花的数据量是真的小，导致得到的结论都不是很严谨的样子。

训练的这些代码和结果之类的，我已经打包上传了，如果有人要复现的话可以看我提供的conda环境配置。
