"""在完整行情上构造正式 Y；清洗仅在调用方完成所需计算后执行。"""

import pandas as pd

def load_data(data_path):
    """股票代码按字符串读取，保留前面的零；日期读取为日期类型。"""
    return pd.read_csv(
        data_path,
        dtype={"股票代码": "string"},
        parse_dates=["日期"],
    )


def calculate_y(data):
    """Y = 第 5 个后续交易日的收盘价 / 次日开盘价 / 5，不足 5 日则为空。"""
    data = data.copy()

    # 1. 用全市场日期建立交易日历，不能用某只股票的行数代表交易日数。
    trading_days = pd.DatetimeIndex(data["日期"].drop_duplicates().sort_values())
    day_numbers = pd.Series(range(len(trading_days)), index=trading_days)
    data["交易日序号"] = data["日期"].map(day_numbers)

    # 2. 固定使用第 5 个后续交易。
    #    序号超出数据末尾时，结束日期自动为空。
    end_day = data["交易日序号"] + 5
    data["次日日期"] = (data["交易日序号"] + 1).map(pd.Series(trading_days))
    data["Y结束日期"] = end_day.map(pd.Series(trading_days))

    # 3. 按“股票代码 + 交易日序号”查价格，保证日期精确对应。
    prices = data[
        ["股票代码", "交易日序号", "复盘后开盘价", "复权后收盘价"]
    ].set_index(["股票代码", "交易日序号"])

    next_day_keys = pd.MultiIndex.from_arrays(
        [data["股票代码"], data["交易日序号"] + 1],
        names=["股票代码", "交易日序号"],
    )
    end_day_keys = pd.MultiIndex.from_arrays(
        [data["股票代码"], end_day],
        names=["股票代码", "交易日序号"],
    )

    data["次日开盘价"] = prices["复盘后开盘价"].reindex(next_day_keys).to_numpy()
    data["第5日收盘价"] = prices["复权后收盘价"].reindex(end_day_keys).to_numpy()

    # 4. 按指定公式计算，不减去 1。退市或数据末尾不足 5 日时，Y 为空。
    #    缺少次日开盘价或第 5 日收盘价时，同样为空，不填充或跳过日期。
    data["Y"] = data["第5日收盘价"] / data["次日开盘价"] / 5
    valid_prices = (data["次日开盘价"] > 0) & (data["第5日收盘价"] > 0)
    data["Y"] = data["Y"].where(valid_prices)

    # 五个原始特征保持不变；新增的价格和日期列只用于检查 Y。
    return data.drop(columns="交易日序号")


def clean_data(data, subset=None):
    """按必需字段清洗；默认保留原评价逻辑的所有列缺失检查。"""
    return data.dropna(subset=subset).copy()


def prepare_data(data_path):
    """先在完整行情上计算正式 Y，保留缺失行供后续目标和规则计算。"""
    return calculate_y(load_data(data_path))
