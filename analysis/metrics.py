"""训练指标与正式 Y 的逐日 IC；不执行拟合、切分或样本筛选。"""

import numpy as np
import pandas as pd

if __package__:
    from .data import clean_data
    from .models import data_loss, objective, predict
else:
    from data import clean_data
    from models import data_loss, objective, predict


def evaluate_training(model, X, y_train, dates, config):
    """训练指标针对 Y_train 计算；改变训练目标后，其尺度也随之改变。"""
    if config["loss"] == "lhalf":
        return evaluate_lhalf_training(model, X, y_train, dates, config)
    predictions = predict(model, X)

    mse = data_loss(y_train, predictions)
    baseline_mse = np.mean((y_train - y_train.mean()) ** 2)
    r_squared = 1 - mse / baseline_mse if baseline_mse > 0 else np.nan

    return {
        "train_mse": mse,
        "train_loss": data_loss(y_train, predictions, kind=config["loss"], dates=dates),
        "train_baseline_mse": baseline_mse,
        "train_r2": r_squared,
        "train_objective": objective(model, X, y_train, dates, config),
    }


def evaluate_lhalf_training(model, X, y_train, dates, config):
    """平滑目标为实际优化损失，未平滑的 mean(sqrt(abs(r))) 单独报告。"""
    predictions = predict(model, X)
    mse = data_loss(y_train, predictions)
    baseline_mse = np.mean((y_train - y_train.mean()) ** 2)
    loss = objective(model, X, y_train, dates, config)
    return {
        "train_mse": mse,
        "train_loss": loss,
        "train_baseline_mse": baseline_mse,
        "train_r2": 1 - mse / baseline_mse if baseline_mse > 0 else np.nan,
        "train_objective": loss,
        "train_lhalf_unsmoothed": np.mean(np.sqrt(np.abs(y_train - predictions))),
    }


def evaluate(model, eval_data, feature_columns):
    """所有实验在同一个正式 Y 上评价，返回预测结果、每日 IC、均值和标准差。"""
    # 也支持独立传入尚未清洗的数据，复用相同的清洗规则。
    eval_data = clean_data(eval_data)
    X = eval_data[feature_columns].to_numpy(dtype=float)
    eval_data["因子值"] = predict(model, X)

    # 2. 每个交易日为一个截面，计算预测值与真实 Y 的 Pearson 相关系数。
    ic_by_date = {}
    for date, section in eval_data.groupby("日期"):
        if (
            len(section) < 2
            or section["因子值"].nunique() < 2
            or section["Y"].nunique() < 2
        ):
            ic_by_date[date] = np.nan
        else:
            ic_by_date[date] = section["因子值"].corr(section["Y"], method="pearson")

    daily_ic = pd.Series(ic_by_date, name="IC", dtype=float)
    daily_ic.index.name = "日期"

    # 3. 每个交易日等权；无法计算 IC 的日期为 NaN，不参与统计。
    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std(ddof=1)
    return eval_data, daily_ic, mean_ic, std_ic


def plot_daily_ic(daily_ic, output_path, *, title="Daily IC", rolling_window=20):
    """保存最终评价的每日 IC、向后滚动均值和逐年均值；不修改输入、不弹窗。"""
    from pathlib import Path
    from textwrap import fill

    from matplotlib import dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not isinstance(rolling_window, int) or isinstance(rolling_window, bool) or rolling_window < 1:
        raise ValueError("rolling_window 必须为正整数。")
    series = daily_ic.copy().astype(float)
    series.index = pd.to_datetime(series.index)
    if series.index.isna().any() or series.index.has_duplicates:
        raise ValueError("每日 IC 的日期不能缺失或重复。")
    series = series.sort_index().replace([np.inf, -np.inf], np.nan)
    # 保留无效日期形成断点；仅使用当前及过去窗口，不作前向填充或居中平滑。
    rolling = series.rolling(rolling_window, min_periods=rolling_window).mean()
    yearly = series.groupby(series.index.year).agg(["mean", "count"])
    valid = series.dropna()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(13, 5.5), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    figure.suptitle(str(title), fontsize=16, fontweight="bold")
    ax.set_title(
        f"Daily Pearson IC  |  valid days: {len(valid)} / {len(series)}"
        + (f"  |  mean: {valid.mean():+.5f}" if len(valid) else "  |  no valid observations"),
        fontsize=10, color="#475569", pad=12,
    )
    ax.axhline(0, color="#64748b", linewidth=0.9, zorder=1)
    if not series.empty:
        ax.plot(series.index, series, color="#94b9db", linewidth=0.8, label="Daily IC", zorder=2)
        ax.plot(series.index, rolling, color="#d97706", linewidth=2.0,
                label=f"Trailing {rolling_window}-observation mean", zorder=4)
        mean_label_used = False
        for number, (year, stats) in enumerate(yearly.iterrows()):
            section = series.loc[series.index.year == year]
            if number % 2 == 1:
                ax.axvspan(section.index.min(), section.index.max(), color="#f1f5f9", zorder=0)
            if stats["count"]:
                ax.hlines(stats["mean"], section.index.min(), section.index.max(),
                          color="#334155", linewidth=1.4, linestyles="--", zorder=3,
                          label=None if mean_label_used else "Calendar-year mean")
                mean_label_used = True
            if number:
                ax.axvline(pd.Timestamp(year=int(year), month=1, day=1), color="#cbd5e1", linewidth=0.8, zorder=1)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.margins(x=0.015)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    if valid.empty:
        ax.text(0.5, 0.5, "No valid daily IC", transform=ax.transAxes, ha="center", color="#64748b")
    # 对称坐标，留出空间使零线上下的变化便于比较；不裁掉极值。
    bound = max(0.1, float(valid.abs().max()) * 1.12) if len(valid) else 0.1
    ax.set_ylim(-bound, bound)
    ax.set_ylabel("Pearson IC")
    ax.set_xlabel("Signal date")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    yearly_text = "   |   ".join(
        f"{year}: {row['mean']:+.5f} (n={int(row['count'])})" if row["count"]
        else f"{year}: no valid IC"
        for year, row in yearly.iterrows()
    )
    figure.supxlabel(fill(yearly_text, width=120), fontsize=10, color="#334155")
    try:
        figure.savefig(output_path, dpi=160, format="png")
    finally:
        figure.clear()
    return output_path
