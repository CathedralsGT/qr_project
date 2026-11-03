# config.py 实验配置手册

在同目录 `config.py` 的 `EXPERIMENTS` 中配置实验。列表中的每个字典代表一项实验，程序按列表顺序运行。

阅读顺序：[Quick Start](#1-quick-start) → [必填参数](#2-必填参数) → [选填参数](#3-选填参数) → [参数联动](#4-参数联动) → [代表性示例](#5-代表性示例) → [Common Pitfalls](#6-common-pitfalls常见误区)。

## 1. Quick Start

### 1.1 只包含必填参数的完整框架

将 `config.py` 中的 `EXPERIMENTS` 设为：

```python
EXPERIMENTS = [
    {
        "name": "my_ols_raw",    # 实验名称
        "sample_rule": "all",   # 选择哪些训练样本
        "target": "raw",        # 如何构造训练标签
        "loss": "mse",          # 用什么误差衡量拟合好坏
        "penalty": "none",      # 是否惩罚模型系数
        "alpha": 0.0,           # 惩罚项强度
        "solver": "ols",        # 使用哪个求解器
    },
]
```

这项实验使用默认五特征，在完整训练期拟合一次 OLS，再计算测试 IC。**最小配置不执行内部验证**；若还需要验证 IC，在字典中加入选填项 `"alphas": [0.0]`。

要增加实验，在列表中追加字典；要只运行几项，暂时移除或注释其他字典。每项实验的 `name` 应唯一。当前没有 `enabled` 字段或按实验名筛选的命令行参数。

### 1.2 运行与查看结果

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m analysis.analysis
```

正常运行会复用匹配的已完成结果。强制重跑当前列表用 `--force`；指定结果目录用 `--results-dir`：

```powershell
.\.venv\Scripts\python.exe -m analysis.analysis --force
.\.venv\Scripts\python.exe -m analysis.analysis --results-dir analysis\results_trial
```

| 输出 | 用途 |
| --- | --- |
| `analysis/results/summary.csv` | 查看当前列表中已完成实验的指标与配置。 |
| `analysis/results/<实验名_编号>/result.json` | 查看完整配置、选定的 alpha、系数和指标。 |
| 同目录 `daily_ic.csv`、`daily_ic.png` | 查看测试期逐日 IC 及其图表。 |
| 同目录 `alpha_search.csv` | 查看该实验的内部验证明细；未验证时为空。 |
| `analysis/alpha_search_results.csv` | 汇总当前列表的验证明细，有明细时导出。 |
| `analysis/train_data.pkl` | 供绘图使用的共同训练数据，不是每项实验最终训练样本的导出。 |

优先看 `validation_ic`（内部验证日均 IC）、`mean_ic`（测试日均 IC）、`n_train`（最终训练行数）和 `n_eval`（测试评价行数）。`std_ic` 为测试逐日 IC 的样本标准差，`valid_ic_days` 为有效评价天数。

`--results-dir` 只切换逐实验结果及 `summary.csv` 的目录；其余两个全局导出文件的路径仍由 `config.py` 顶部常量决定。移除实验配置不会删除磁盘上的历史结果。

## 2. 必填参数

每个实验都必须包含以下七个字段。参数值区分大小写，字符串应使用下文给出的拼写。

### 2.1 `name`：标识一项实验

取值为实验名称字符串，建议使用英文、数字和下划线，例如 `"ols_5f_fill0"`。

名称用于结果展示、文件夹命名和返回结果的字典键，也参与结果缓存编号。同一列表中应保持唯一，重名可能覆盖按名称索引的模型或逐日 IC。

### 2.2 `sample_rule`：控制选取哪些样本作为实际训练样本的规则

| 可选值 | 具体行为 |
| --- | --- |
| `"all"` | 在样本选择阶段选中当前拟合期的所有行，不额外筛选。之后仍执行输入填补、缺失清洗及标签有效性检查，因此不保证所有行最终进入模型。 |
| `"complete_5f"` | 只选中全局五个特征均为有限数值的行。即使模型只使用前三个特征，换手率或波动率缺失的行也会被排除。 |

该规则同时作用于内部拟合和最终重训，**不用于筛选验证、测试股票**。`complete_5f` 检查的是填补前的特征；五个特征来自全局 `FEATURE_COLUMNS`。

### 2.3 `target`：控制训练标签 `Y_train` 的构造方式

| 可选值 | 具体行为 |
| --- | --- |
| `"raw"` | 直接使用正式 `Y` 作为训练标签；跨越本次拟合截止边界的标签置为空。 |
| `"daily_zscore"` | 将当日有效 `Y` 转为 `(Y - 当日均值) / 当日标准差`，标准差使用总体口径 `ddof=0`。有效标签少于两只、标准差为零或非有限的日期，其训练标签置为空。 |

两种方式都先隔离跨界标签。日标准化在样本筛选和特征清洗之前计算，因此某行即使后来因特征缺失被删除，也可能参与当日标签统计。最终保留的训练截面不保证仍然均值为 0、方差为 1。

正式 `Y` 由行情生成，目前定义为第 5 个后续市场交易日收盘价除以次日开盘价，再除以 5，不减 1。**验证、测试始终针对正式 `Y` 计算每日 Pearson IC**，不随 `target` 改变。

### 2.4 `loss`：控制模型拟合时最小化的数据误差

下面记残差为 `r = Y_train - 预测值`。

| 可选值 | 具体行为 |
| --- | --- |
| `"mse"` | 最小化所有训练行的平均平方误差 `mean(r²)`；每行权重相同。 |
| `"mae"` | 最小化所有训练行的平均绝对误差 `mean(abs(r))`。 |
| `"l1"` | `"mae"` 的兼容别名，行为完全相同。 |
| `"lhalf"` | 最小化平滑残差 L0.5 目标 `mean((r² + smooth_eps²)^(1/4))`，使用 IRLS 求局部解。它作用于残差，不是系数正则项。 |
| `"daily_mse"` | 每天先计算平均平方误差，再对训练日期取等权平均；每个日期的总权重相同。 |
| `"daily_centered_mse"` | 每天先对残差去均值，再计算日内平均平方误差，最后对日期等权平均。求解时等价于对当日 X 和 Y_train 分别去均值后做按日加权最小二乘。 |

按日损失的每日样本数由**最终清洗后的训练行**计算。中心化损失下，只有一个样本的日期贡献为 0，但仍计入日期平均。

`loss` 需要与 `solver`、`penalty` 配套，合法组合见第 4.1 节。

### 2.5 `penalty`：控制对斜率系数施加哪种惩罚

所有惩罚只作用于 `beta`，不惩罚截距。

| 可选值 | 具体行为 |
| --- | --- |
| `"none"` | 不添加系数惩罚，惩罚项为 0。 |
| `"ridge"` | 添加 L2 平方惩罚 `sum(beta²)`。 |
| `"lasso"` | 添加 L1 惩罚 `sum(abs(beta))`，可能使部分系数为 0。 |

总训练目标为 `数据损失 + alpha × 惩罚项`。目前没有 `"l1"`、`"l2"` 或 `"elastic"` 这些 penalty 取值。

### 2.6 `alpha`：控制惩罚项强度

取值为单个有限、非负数，例如 `0.0`、`0.001`。具体求解器可能进一步要求只能为 0。

省略 `alphas` 时，模型直接使用这个值。有 `alphas` 时，拟合使用候选值及最终选定值，但配置中仍需保留 `alpha` 字段，通常填写 `0.0`。

本项目的 MSE 定义不带 `1/2`，Ridge 惩罚也不带 `1/2`。与其他库比较正则强度时，先对齐目标函数定义，不能仅比较 alpha 数字。

### 2.7 `solver`：控制使用哪个算法求解模型系数

| 可选值 | 具体行为 |
| --- | --- |
| `"ols"` | 求普通最小二乘解。 |
| `"ridge"` | 求 MSE + L2 惩罚的解；alpha=0 时回退为 OLS。 |
| `"lasso"` | 用坐标下降求 MSE + L1 惩罚的解；alpha=0 时回退为 OLS。 |
| `"admm"` | 求 MAE，可搭配无惩罚、L1 或 L2 惩罚。 |
| `"irls"` | 求平滑残差 L0.5 的局部解；从 OLS 和常数中位数两个初值分别迭代，按训练平滑损失选较优结果。两个初值都须收敛，不保证全局最优。 |
| `"daily_wls"` | 求每日等权 MSE 或每日等权的中心化 MSE，使用加权最小二乘。 |

所有求解器最终返回同一线性预测形式 `X @ beta + b`。除 `daily_centered_mse` 将不可识别的截距固定为 0 外，其余方法均估计截距。

**配置字段名是 `solver`，没有 `model` 字段。** 改变 penalty 不会自动切换 solver。

## 3. 选填参数

以下字段按需要添加，不写时使用本节说明的默认行为。

### 3.1 `alphas`：控制是否做内部验证，以及搜索哪些正则强度

默认省略，也可设为 `None`：不做内部验证，直接以 `alpha` 在完整训练期拟合。

提供时应为非空、一维的非负有限数候选列表，并满足所选求解器的要求。例如：

| 写法 | 具体行为 |
| --- | --- |
| `"alphas": [0.0]` | 对 alpha=0 验证一次，再在完整训练期重训；适合给 OLS 等固定模型补验证 IC。 |
| `"alphas": [0.001]` | 对 alpha=0.001 验证一次，再重训；仅适用于支持该值的求解器。 |
| `"alphas": [0.0, 0.001, 0.01]` | 逐个验证候选，以验证日均 IC 最大者作为最终 alpha，再重训。 |

候选按输入顺序去重，验证 IC 完全相同时选最先出现者。它只搜索 alpha，不搜索 target、loss、填充值或 smooth_eps；比较这些参数要写不同实验。

空列表、负数、NaN、无穷大不合法。收敛失败或无法计算 IC 的候选会记录状态；如果没有任何有效验证 IC，实验失败。

### 3.2 `feature_columns`：控制模型使用哪些输入及其顺序

默认使用全局五特征：

```python
["原始股价", "成交金额", "总市值", "换手率", "二十日波动率"]
```

例如三特征实验填写 `"feature_columns": ["原始股价", "成交金额", "总市值"]`。列顺序就是结果中 beta 的顺序。

项目内应使用上述列的非空、不重复子集。训练清洗只要求实际输入及标签有效，因此减少输入列也可能增加训练行数；需要同训练样本对照时，配合 `sample_rule="complete_5f"`。

单项实验应设置自己的 `feature_columns`，不要为它修改全局 `FEATURE_COLUMNS`，后者还影响其他默认实验、完整样本筛选与数据导出。

### 3.3 `evaluation_pool`：控制验证、测试时评价哪些股票行

| 可选值 | 具体行为 |
| --- | --- |
| `"common_5f"`（默认） | 沿用完整评价表的缺失检查。当前基础数据下对应原五特征共同评价池；如果添加了辅助列，辅助列缺失也可能导致删行。 |
| `"selected_features"` | 先只保留股票代码、日期、所选输入和正式 Y，再删除这些列存在缺失的行；未使用列的缺失不影响评价。 |
| `"all_3f"` | `"selected_features"` 的兼容别名，不会自动把模型输入改成三个特征。 |

它同时作用于验证和测试，**不控制训练样本，也不填补评价数据**。三特征使用 `selected_features` 时可以保留波动率缺失行；五特征仍然需要波动率有效。

### 3.4 `train_feature_fill`：控制如何填补训练输入的缺失值

默认省略或 `{}`，不填补。提供时必须是“本实验输入列 → 有限数值常数”的字典，例如 `"train_feature_fill": {"二十日波动率": 0.0}`。

指定列中的 NaN 会被填为给定常数；训练清洗先将正负无穷转为 NaN，因此这些值也会被填补。已有有限值（包括 0）保持不变。未指定的输入列以及标签、日期均不会被填补。

该操作在样本筛选之后、最终缺失删行之前执行，同时用于内部拟合与最终重训，不修改共享原始数据，也不用于验证、测试。

字典键必须属于当前 `feature_columns`，值不能为字符串、布尔值、NaN 或无穷大；`None` 也不是合法配置。当前只支持常数填补，没有中位数、前向填充或历史波动率重算模式。

### 3.5 `max_iter`：控制迭代次数上限

填写正整数，仅用于迭代求解器。

| 求解器 | 默认值 | 含义 |
| --- | --- | --- |
| `lasso`，alpha > 0 | 10000 | 坐标下降迭代上限。 |
| `admm` | 10000 | ADMM 迭代上限。 |
| `irls` | 1000 | 每个初值各自的 IRLS 迭代上限。 |

`ols`、`ridge`、`daily_wls` 不使用该参数；Lasso 的 alpha=0 分支也不使用。达到上限但尚未收敛会报告失败，不作为成功结果返回。

### 3.6 `tol`：控制迭代收敛容差

填写正的有限数。数值越小，要求越严格，但不同求解器的判据不同。

| 求解器 | 默认值 | 判断内容 |
| --- | --- | --- |
| `lasso`，alpha > 0 | `1e-10` | 系数满足最优性条件的误差。 |
| `admm` | `1e-7` | 原始约束误差及对偶残差。 |
| `irls` | `1e-6` | 内部归一化参数的相对更新步长。 |

其他求解器不使用该参数。不要将不同求解器的同一 tol 数值理解为相同拟合精度。

### 3.7 `rho`：控制 ADMM 的数值求解尺度

仅供 `solver="admm"` 使用。默认省略或 `None`，由当前拟合标签自动设置：`1 / max(median(abs(y - median(y))), 1e-6)`。

也可手动填写正的有限数。rho 影响数值迭代，不是模型正则强度；正则强度仍由 alpha 控制。

### 3.8 `smooth_eps`：控制残差 L0.5 的平滑尺度

仅供 `loss="lhalf", solver="irls"` 使用，默认 `1e-4`，必须为正的有限数，单位与 `Y_train` 相同。

实际目标是 `mean((r² + smooth_eps²)^(1/4))`。数值越小越接近未平滑的 `mean(abs(r)^0.5)`，但可能更难收敛。它不是正则强度，也不能设为 0。

切换 raw 与 daily_zscore 后要重新考虑平滑尺度。多个 smooth_eps 要用多个实验比较，不能放进 `alphas`。

## 4. 参数联动

### 4.1 `loss`、`penalty`、`solver` 必须成套配置

从下表选择一行；不要把不同方法的字段任意拼接。表中 alpha 的要求也适用于候选 `alphas`。

| 方法 | loss | penalty | solver | alpha 要求 |
| --- | --- | --- | --- | --- |
| OLS | `mse` | `none` | `ols` | 只能为 0。 |
| Ridge | `mse` | `ridge` | `ridge` | 非负有限数。 |
| Lasso | `mse` | `lasso` | `lasso` | 非负有限数。 |
| MAE，无正则 | `mae` / `l1` | `none` | `admm` | 建议为 0；当前接受正值，但不产生正则作用。 |
| MAE + L2 | `mae` / `l1` | `ridge` | `admm` | 非负有限数。 |
| MAE + L1 | `mae` / `l1` | `lasso` | `admm` | 非负有限数。 |
| 平滑残差 L0.5 | `lhalf` | `none` | `irls` | 只能为 0。 |
| 每日等权 MSE | `daily_mse` | `none` | `daily_wls` | 只能为 0。 |
| 每日等权、截面中心化 MSE | `daily_centered_mse` | `none` | `daily_wls` | 只能为 0。 |

这些组合均可使用两种 target。`daily_zscore` 只改变标签，不自动改变日期权重或使预测中心化；后两者由 loss 决定。改变标签尺度后，应重新验证正则强度和平滑尺度。

### 4.2 样本规则、填补与评价池的执行顺序

每次拟合都独立执行：

1. 按日期取得当前拟合期，将越过截止边界的未来标签置空。
2. 按 target 计算 Y_train，再按 sample_rule 筛选行。
3. 将实际输入与标签中的非有限值转为 NaN，按 train_feature_fill 填补指定输入。
4. 删除日期、实际输入、Y 或 Y_train 仍缺失的行，交给求解器。

因此，`complete_5f` 会在填补前删除波动率缺失行；要通过填补保留这些行，应使用 `sample_rule="all"`。三特征实验不能配置填补未使用的波动率列。

验证和测试另行按 evaluation_pool 取得评价视图，再用正式 Y 计算 IC。比较训练方法时，先固定评价池；改变训练样本与扩大评价股票池是两个不同的实验操作。

### 4.3 `alpha`、`alphas` 与时间窗口

时间窗口是整组实验共用的全局配置，不是单个实验的字段：

```python
SPLIT_CONFIG = {
    "train_start": "2012-01-01",
    "train_end": "2018-12-31",
    "test_start": "2019-01-01",
    "test_end": "2020-12-31",
    "validation_start": "2018-01-01",
}
```

没有 alphas 时，直接用完整训练期和固定 alpha 拟合。有 alphas 时，先用 validation_start 之前的数据拟合候选，在之后的训练期数据上验证；选定 alpha 后，用完整训练期重新拟合，再评价测试期。

外层信号日期筛选包含起止日期；内部拟合的标签结束日期必须严格早于 validation_start，最终重训的标签结束日期必须严格早于 test_start。请使训练、测试不重叠，且内部拟合和验证都有数据。数据末尾缺少未来价格的标签无法用于评价。

**验证 IC 来自内部拟合模型，测试 IC 来自完整训练期重训模型。** 程序只在每项实验内部选 alpha，不自动跨实验选择最终方法。修改时间窗口应编辑 SPLIT_CONFIG；无需单独修改其兼容别名 VALIDATION_START。

## 5. 代表性示例

以下代码追加到 `EXPERIMENTS` 定义之后，也可以将其中的字典直接放进列表。示例名称使用 `manual_` 前缀，避免与现有默认实验重名；重复运行同一段追加代码仍会产生重复名称。

### 5.1 区分减少特征、扩大训练样本和缺失填补

这组包含五特征基准、同样本三特征、扩大样本三特征、五特征填 0，四者固定共同评价池。

```python
for name, options in [
    ("manual_5f_baseline", {}),
    ("manual_3f_same_rows", {
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "sample_rule": "complete_5f",
    }),
    ("manual_3f_all", {
        "feature_columns": ["原始股价", "成交金额", "总市值"],
    }),
    ("manual_5f_fill0", {
        "train_feature_fill": {"二十日波动率": 0.0},
    }),
]:
    EXPERIMENTS.append({
        "name": name,
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "ols",
        "alphas": [0.0],
        "evaluation_pool": "common_5f",
        **options,
    })
```

第一项与第二项比较相同训练行上减少特征的影响；第二项与第三项比较三特征下扩样本的影响。当前数据只有波动率缺失时，第三项与第四项可比较扩大样本后保留五特征的效果；其他数据应先核对训练键和行数。

### 5.2 Ridge 与 Lasso：搜索正则强度

```python
for method, candidates in [
    ("ridge", [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
    ("lasso", [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
]:
    EXPERIMENTS.append({
        "name": f"manual_{method}_raw",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": method,
        "alpha": 0.0,
        "solver": method,
        "alphas": candidates,
    })
```

候选网格是起点，不代表最优范围。要比较标准化标签，另建实验并修改 target，再检查候选范围。

### 5.3 MAE：更换数据损失，保持无正则

```python
EXPERIMENTS.append({
    "name": "manual_mae_raw",
    "sample_rule": "all",
    "target": "raw",
    "loss": "mae",
    "penalty": "none",
    "alpha": 0.0,
    "solver": "admm",
    "alphas": [0.0],
    "max_iter": 20000,
    "tol": 1e-6,
})
```

若要试 MAE + L1 正则，换一个 name，将 penalty 改为 `"lasso"`，alphas 改为例如 `[0.0, 1e-5, 1e-4]`，**solver 仍为 `"admm"`**。MAE + L2 同理使用 penalty=`"ridge"`。

### 5.4 日标准化标签：逐行损失、每日等权、截面中心化

```python
for loss, solver in [
    ("mse", "ols"),
    ("daily_mse", "daily_wls"),
    ("daily_centered_mse", "daily_wls"),
]:
    EXPERIMENTS.append({
        "name": f"manual_zscore_{loss}",
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": loss,
        "penalty": "none",
        "alpha": 0.0,
        "solver": solver,
        "alphas": [0.0],
        "evaluation_pool": "common_5f",
    })
```

三项固定标签定义、输入和样本规则，依次改变日期权重和中心化处理。日期由流程自动传递，无需在实验字典中填写 dates。

### 5.5 平滑残差 L0.5：指定平滑程度和迭代参数

```python
EXPERIMENTS.append({
    "name": "manual_lhalf_raw",
    "sample_rule": "all",
    "target": "raw",
    "loss": "lhalf",
    "penalty": "none",
    "alpha": 0.0,
    "solver": "irls",
    "alphas": [0.0],
    "smooth_eps": 1e-4,
    "max_iter": 1000,
    "tol": 1e-6,
})
```

`train_loss` 和 `train_objective` 报告实际平滑目标，`train_lhalf_unsmoothed` 单独报告未平滑的 `mean(abs(r)^0.5)`。初值按训练损失选，不按验证或测试 IC 选；不能把局部求解结果视为全局最优。

## 6. Common Pitfalls：常见误区

| 现象或填法 | 原因与处理 |
| --- | --- |
| 只改 loss 或 penalty，保留原 solver | 三者不会自动联动。按第 4.1 节整行选择；例如 MAE + Lasso 仍用 ADMM。 |
| `loss="l1"` 被当成 L1 正则 | 它是 MAE 数据损失的别名；系数 L1 正则应写 `penalty="lasso"`。 |
| 给 daily_wls 或 irls 配正则项、正 alpha | 当前不支持；两者均要求 penalty=`none`、alpha=0。 |
| `alphas=[]` 或给 OLS 填正的候选 | 空列表不合法，OLS 候选只能为 0。跳过验证应省略 alphas 或设为 None。 |
| `validation_ic` 是 NaN | 未设置 alphas 时属于预期行为，不表示测试失败。固定模型需要验证时填 `[0.0]`。 |
| 搜索表出现 `no_valid_ic` | 预测或标签为常数、有效股票过少等情况使 IC 不可计算。检查数据和正则强度；所有候选均无有效 IC 时会报错。 |
| 填 0 后训练行数没有增加 | 检查是否使用了 complete_5f，或该行还有其他输入、标签缺失。填补不能恢复已被样本规则删掉的行。 |
| 以为填 0 会替换所有原有 0，或恢复真实波动率 | 只填指定列的缺失值；已有有限值不变。给定特征已标准化，填 0 不等于实际波动率为零。 |
| 三特征实验填写波动率填补，或把 train_feature_fill 设为 None | 填补键必须属于实际输入。关闭填补应省略字段或用 `{}`。 |
| 训练填补后以为验证、测试也会扩样本 | train_feature_fill 只用于训练。评价池单独由 evaluation_pool 控制，五特征评价仍会排除波动率缺失行。 |
| 减少特征后 IC 提升，就归因于特征更好 | 训练行数也可能增加。先用 complete_5f 做同样本对照，再检查 n_train、n_eval 和评价池。 |
| daily_zscore 被当成按日等权或预测中心化 | 它只改变标签。日期等权和中心化分别由相应 loss 控制。 |
| 改了 target 后直接比较 train_mse、train_r2，或认为训练损失越小测试 IC 就越高 | 标签尺度、参考方差、样本构成或目标函数可能已变化。训练指标用于诊断，泛化比较看相同评价口径下的 IC。 |
| 按日损失或 L0.5 实验只看 train_mse | train_mse 是普通逐行平方误差；实际优化项看 train_loss，总目标看 train_objective。 |
| `smooth_eps=0`，或将 epsilon 放入 alphas | smooth_eps 必须为正；alphas 只搜索正则强度。改变 epsilon 应建立不同实验。 |
| Lasso、MAE 或 IRLS 不收敛 | 检查数据尺度，增加适用求解器的 max_iter 或适当放宽 tol；ADMM 可调整 rho，IRLS 可调整 smooth_eps。 |
| 添加 `model`、`fit_intercept`、`enabled`、`daily_centered` 等字段 | 当前没有这些逐实验接口。字段可能被忽略，不能靠新增键启用功能。 |
| 在单个实验中写 train_start 或 validation_start | 不会覆盖全局窗口；应修改 SPLIT_CONFIG。 |
| 使用未实现的取值，例如 target=`rank`、loss=`huber` | 只改字符串不会增加算法，需要扩展对应实现。 |
| 误以为所有错拼字段都会立即报错 | 当前没有统一的严格字段校验。请按本手册填写；未知键可能被忽略。 |
| 把 train_data.pkl 当成填补后实际训练数据 | 它是独立导出的共同绘图数据，不代表每个实验最终使用的行或值。 |

修改配置后，先核对实验名称、样本数、选定 alpha 和评价池，再解释指标差异。
