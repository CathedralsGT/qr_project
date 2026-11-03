"""当前拟合期内依次计算 Y_train、选取标志，筛选后填补指定输入并清洗。"""

from numbers import Real

import numpy as np
import pandas as pd

if __package__:
    from .config import FEATURE_COLUMNS
    from .data import clean_data
else:
    from config import FEATURE_COLUMNS
    from data import clean_data


def make_selection_flag(data, rule="all", *, cutoff):
    """返回逐行选取标志，不删行；新增规则须将依赖 cutoff 及之后数据的结果置空。"""
    if rule == "all":
        return pd.Series(True, index=data.index)
    if rule == "complete_5f":
        values = data[FEATURE_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
        return pd.Series(np.isfinite(values).all(axis=1), index=data.index)
    raise ValueError(f"尚未实现的训练样本规则：{rule}")


def make_training_target(data, target="raw", *, cutoff):
    """先在未筛选的拟合期计算目标；新增目标须将越过 cutoff 的依赖置空。"""
    if target == "raw":
        return data["Y"].where(data["Y结束日期"] < cutoff)
    if target == "daily_zscore":
        # 先隔离越界和非有限标签；特征缺失及稍后被筛除的行仍属于参考截面。
        y = data["Y"].where(data["Y结束日期"] < cutoff)
        y = y.replace([np.inf, -np.inf], np.nan)
        grouped = y.groupby(data["日期"], sort=False)
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)
        count = grouped.transform("count")
        valid = (count >= 2) & (std > 0) & np.isfinite(std)
        return (y - mean) / std.where(valid)
    raise ValueError(f"尚未实现的训练目标：{target}")


def validate_feature_fill(fill_values, feature_columns):
    """仅允许填补本实验使用的输入列，填充值必须是有限数值常数。"""
    if not isinstance(fill_values, dict):
        raise ValueError("train_feature_fill 必须是输入列到有限数值常数的字典。")
    for column, value in fill_values.items():
        if column not in feature_columns or column in ("Y", "Y_train", "日期", "股票代码"):
            raise ValueError(f"train_feature_fill 只能指定本实验使用的输入列：{column}")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"train_feature_fill 的填充值必须是有限数值常数：{column}")


def prepare_training_data(base_train, config, *, cutoff, feature_columns):
    """边界检查/置空 → Y_train → 选取标志 → 筛选 → 输入填补 → 必需字段清洗。"""
    fill_values = config.get("train_feature_fill", {})
    validate_feature_fill(fill_values, feature_columns)
    train_data = base_train.copy()
    if (train_data["日期"] >= cutoff).any():
        raise ValueError("请先按日期划分拟合集，信号日期必须严格早于 cutoff。")

    # Y 已在完整行情上计算。将预先连接的未来值置空后，再用于任何目标/规则统计。
    # 不删除越界行，保留完整交易日期与股票记录；也不修改共同训练数据。
    for value_column, end_column in (
        ("Y", "Y结束日期"),
        ("第5日收盘价", "Y结束日期"),
        ("次日开盘价", "次日日期"),
    ):
        if value_column in train_data:
            train_data[value_column] = train_data[value_column].where(train_data[end_column] < cutoff)

    train_data["Y_train"] = make_training_target(train_data, config["target"], cutoff=cutoff)
    train_data["selected"] = make_selection_flag(train_data, config["sample_rule"], cutoff=cutoff)
    train_data = train_data.loc[train_data["selected"].eq(True)].copy()

    required_columns = list(feature_columns) + ["Y", "Y_train"]
    train_data[required_columns] = train_data[required_columns].replace([np.inf, -np.inf], np.nan)
    if fill_values:
        train_data = train_data.fillna(fill_values)
    return clean_data(train_data, subset=["日期"] + required_columns)


def as_training_arrays(train_data, feature_columns):
    """在数据边界按明确的特征顺序转换数组；模型层无需了解列名。"""
    if train_data.empty:
        raise ValueError("训练样本筛选或目标变换后，拟合集为空。")
    X = train_data[feature_columns].to_numpy(dtype=float)
    y_train = train_data["Y_train"].to_numpy(dtype=float)
    return X, y_train
