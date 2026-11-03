"""实验编排和命令行入口；核心计算与打印、文件导出分开。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 同时支持 python analysis/analysis.py 与 python -m analysis.analysis。
# 也保留绘图脚本 from analysis import FEATURE_COLUMNS, TRAIN_OUTPUT_PATH 的入口。
if __package__:
    from .config import (
        ALPHA_SEARCH_OUTPUT_PATH, DATA_PATH, EXPERIMENTS, FEATURE_COLUMNS,
        SPLIT_CONFIG, TRAIN_OUTPUT_PATH, VALIDATION_START,
        get_model_config, get_training_config,
    )
    from .data import clean_data, prepare_data
    from .metrics import evaluate, evaluate_training
    from .models import fit
    from .results import build_context, experiment_key, export_summary, load_completed_result, save_result, save_ic_plot
    from .selection import select_alpha
    from .splits import split_data, validation_split
    from .training_data import as_training_arrays, prepare_training_data
else:
    from config import (
        ALPHA_SEARCH_OUTPUT_PATH, DATA_PATH, EXPERIMENTS, FEATURE_COLUMNS,
        SPLIT_CONFIG, TRAIN_OUTPUT_PATH, VALIDATION_START,
        get_model_config, get_training_config,
    )
    from data import clean_data, prepare_data
    from metrics import evaluate, evaluate_training
    from models import fit
    from results import build_context, experiment_key, export_summary, load_completed_result, save_result, save_ic_plot
    from selection import select_alpha
    from splits import split_data, validation_split
    from training_data import as_training_arrays, prepare_training_data


RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class ExperimentResult:
    """一次实验的完整计算结果；不持有完整的逐股预测表。"""

    selected_config: dict
    model: tuple
    metrics: dict
    daily_ic: pd.Series
    search_results: pd.DataFrame


@dataclass
class ExperimentRun:
    """一组实验的共享切分信息、结果及待导出的搜索明细。"""

    summary: pd.DataFrame
    models: dict
    daily_ics: dict
    experiments: list
    search_summary: pd.DataFrame
    n_tuning: int
    n_validation: int


def _evaluation_view(data, feature_columns, pool):
    """按同一口径准备验证和测试视图，不改变原始数据或正式 Y。"""
    if pool == "common_5f":
        return data
    if pool in ("selected_features", "all_3f"):
        if data is None:
            return None
        columns = ["股票代码", "日期"] + list(feature_columns) + ["Y"]
        return data[columns].copy()
    raise ValueError(f"尚未实现的评价股票池：{pool}")


def run_experiment(
    base_train, eval_data, config, *, tuning_train=None,
    validation_data=None, validation_start=None, test_start=None,
    feature_columns=FEATURE_COLUMNS,
):
    """先在内部拟合集选参数，再用完整训练期重新拟合；不打印或写文件。"""
    if "feature_columns" in config:
        feature_columns = config["feature_columns"]
    if "evaluation_pool" in config:
        validation_data = _evaluation_view(validation_data, feature_columns, config["evaluation_pool"])
        eval_data = _evaluation_view(eval_data, feature_columns, config["evaluation_pool"])
    training_config = get_training_config(config)
    model_config = get_model_config(config)
    if validation_start is None:
        validation_start = SPLIT_CONFIG["validation_start"]
    if test_start is None:
        test_start = SPLIT_CONFIG["test_start"]
    alphas = config.get("alphas")
    X_fit, y_fit, dates_fit = None, None, None
    if alphas is not None:
        # 过于保险的assertion，毫无必要
        if tuning_train is None or validation_data is None or tuning_train.empty or validation_data.empty:
            raise ValueError("请先调用 validation_split，提供非空的拟合集和验证集。")
        # 真正的操作部分：如果有多个alpha，则说明需要用 validation set 选择最佳 alpha ，所以先把tuning部分整理出来
        tuning_data = prepare_training_data(
            tuning_train, training_config, cutoff=validation_start, feature_columns=feature_columns,
        )
        X_fit, y_fit = as_training_arrays(tuning_data, feature_columns)
        dates_fit = tuning_data["日期"].to_numpy()

    selected_model_config, search_results = select_alpha(
        X_fit, y_fit, validation_data, model_config, alphas,
        feature_columns=feature_columns,
        dates_fit=dates_fit,
    )
    selected_config = config.copy()
    selected_config.pop("alphas", None)
    selected_config.update(selected_model_config)

    # 内部验证完成后，包含验证年份的完整训练期重新构造 Y_train 和拟合。
    train_data = prepare_training_data(
        base_train, training_config, cutoff=test_start, feature_columns=feature_columns,
    )
    X, y_train = as_training_arrays(train_data, feature_columns)
    model = fit(X, y_train, selected_model_config, dates=train_data["日期"].to_numpy())
    train_metrics = evaluate_training(
        model, X, y_train, train_data["日期"].to_numpy(), selected_model_config,
    )
    test_predictions, daily_ic, mean_ic, std_ic = evaluate(model, eval_data, feature_columns)
    metrics = {
        "name": config["name"],
        "alpha": selected_model_config["alpha"],
        "validation_ic": search_results["validation_ic"].max() if not search_results.empty else np.nan,
        "n_train": len(train_data),
        "n_eval": len(test_predictions),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "valid_ic_days": int(daily_ic.notna().sum()),
    }
    metrics.update(train_metrics)
    return ExperimentResult(selected_config, model, metrics, daily_ic, search_results)


def run_experiments(
    base_train, eval_data, experiments, *, validation_start=None, test_start=None,
    feature_columns=FEATURE_COLUMNS, on_result=None,
):
    """共享一次内部时间切分；默认只计算，可由 on_result 在每次完成后保存结果。"""
    if validation_start is None:
        validation_start = SPLIT_CONFIG["validation_start"]
    if test_start is None:
        test_start = SPLIT_CONFIG["test_start"]
    experiments = list(experiments)

    tuning_train, validation_data = None, None
    # 如果至少一个实验配置有 alphas，就将 base_train 按验证起点切分一次，得到共享的 tuning_train 和 validation_data
    if any(config.get("alphas") is not None for config in experiments):
        tuning_train, validation_data = validation_split(base_train, validation_start)

    results = []
    search_tables = []
    # 遍历每份实验配置，调用一次 run_experiment()
    for config in experiments:
        result = run_experiment(
            base_train, eval_data, config, tuning_train=tuning_train,
            validation_data=validation_data, validation_start=validation_start,
            test_start=test_start, feature_columns=feature_columns,
        )
        results.append(result)
        if on_result is not None:
            on_result(config, result)
        if not result.search_results.empty:
            search_tables.append(result.search_results.assign(experiment=config["name"]))

    # 将返回的单实验结果整理成汇总表、模型字典、逐日 IC 字典和参数搜索明细。
    return ExperimentRun(
        summary=pd.DataFrame([result.metrics for result in results]),
        models={result.selected_config["name"]: result.model for result in results},
        daily_ics={result.selected_config["name"]: result.daily_ic for result in results},
        experiments=results,
        search_summary=pd.concat(search_tables, ignore_index=True) if search_tables else pd.DataFrame(),
        n_tuning=len(tuning_train) if tuning_train is not None else 0,
        n_validation=len(validation_data) if validation_data is not None else 0,
    )


def export_training_data(base_train, output_path, feature_columns):
    """独立清洗绘图所需字段后导出，不修改用于目标/规则计算的训练期数据。"""
    output_columns = ["股票代码", "日期"] + list(feature_columns) + ["Y"]
    clean_data(base_train[output_columns]).to_pickle(output_path)


def export_search_results(search_summary, output_path):
    """只有发生参数搜索时才导出明细。"""
    if not search_summary.empty:
        search_summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def print_experiment_result(result, feature_columns):
    """集中呈现实验结果，不执行拟合或改变计算结果。"""
    config = result.selected_config
    feature_columns = config.get("feature_columns", feature_columns)
    metrics = result.metrics
    if not result.search_results.empty:
        print(f"\n{config['name']} 的 alpha 验证结果：")
        print(result.search_results.to_string(index=False))
        print(f"选定 alpha = {config['alpha']:g}；验证日均 IC = {metrics['validation_ic']:.8f}")

    beta, b = result.model
    print(f"\n实验：{config['name']} | 训练目标：{config['target']} | alpha={config['alpha']:g}")
    print(f"用于拟合的训练集：{metrics['n_train']:,} 行")
    print(f"系数顺序：{feature_columns}")
    print(f"评价股票池：{config.get('evaluation_pool', 'common_5f')}")
    print(f"训练输入缺失填补：{config.get('train_feature_fill', {})}")
    print(f"beta = {beta}")
    print(f"截距 b = {b:.12f}")
    print(f"训练目标平均平方残差（MSE）= {metrics['train_mse']:.12e}")
    print(f"训练数据损失（{config['loss']}）= {metrics['train_loss']:.12e}")
    if config["loss"] == "lhalf":
        print(f"L0.5 平滑参数 smooth_eps = {config.get('smooth_eps', 1e-4):g}；实际优化平滑近似，无全局最优保证。")
        print(f"未平滑的 mean(|残差|^0.5) = {metrics['train_lhalf_unsmoothed']:.12e}")
    if config["loss"] in ("daily_mse", "daily_centered_mse"):
        print("实际优化目标见训练数据损失；MSE、均值基准与 R^2 为未加权、未中心化的辅助指标。")
    print(f"均值预测基准的 MSE = {metrics['train_baseline_mse']:.12e}")
    print(f"训练目标 R^2 = {metrics['train_r2']:.8f}")
    print(f"训练总目标（含惩罚项）= {metrics['train_objective']:.12e}")
    print(f"IC 有效交易日数：{metrics['valid_ic_days']} / {len(result.daily_ic)}")
    print(f"测试集日均 IC = {metrics['mean_ic']:.8f}")
    print(f"测试集 IC 标准差 = {metrics['std_ic']:.8f}")


def main(*, results_dir=None, force=False):
    """先查完成记录；仅为未命中实验准备数据，计算流程保持原样。"""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    context = build_context(DATA_PATH, SPLIT_CONFIG, FEATURE_COLUMNS)
    completed = {}
    pending = []
    for config in EXPERIMENTS:
        key = experiment_key(config, context)
        result = None if force else load_completed_result(results_dir, config, context, ExperimentResult)
        if result is None:
            pending.append(config)
        else:
            completed[key] = result
            save_ic_plot(results_dir, config, context, result)
            print(f"复用已完成实验：{config['name']}")

    def current_results():
        ordered = []
        for config in EXPERIMENTS:
            result = completed.get(experiment_key(config, context))
            if result is not None:
                ordered.append(result)
        return ordered

    def save_finished(config, result):
        save_result(results_dir, config, context, result)
        completed[experiment_key(config, context)] = result
        export_summary(results_dir, current_results())
        plot_path = save_ic_plot(results_dir, config, context, result, overwrite=True)
        print(f"实验完成并已保存：{config['name']}")
        print(f"每日 IC 图：{plot_path}")

    export_summary(results_dir, current_results())
    if pending:
        print(f"需要运行 {len(pending)} 个实验，准备一次共享数据。")
        data = prepare_data(DATA_PATH)
        base_train, eval_data = split_data(
            data,
            train_start=SPLIT_CONFIG["train_start"], train_end=SPLIT_CONFIG["train_end"],
            test_start=SPLIT_CONFIG["test_start"], test_end=SPLIT_CONFIG["test_end"],
        )
        del data
        print(f"训练期参考数据（筛选和清洗前）：{len(base_train):,} 行")
        print(f"测试期数据（评价时清洗）：{len(eval_data):,} 行")
        export_training_data(base_train, TRAIN_OUTPUT_PATH, FEATURE_COLUMNS)
        print(f"训练数据已导出：{TRAIN_OUTPUT_PATH}")

        run = run_experiments(
            base_train, eval_data, pending,
            validation_start=SPLIT_CONFIG["validation_start"], test_start=SPLIT_CONFIG["test_start"],
            feature_columns=FEATURE_COLUMNS, on_result=save_finished,
        )
        if run.n_validation:
            print(f"\n参数选择：{SPLIT_CONFIG['validation_start']} 之前拟合，之后的训练期数据验证")
            print(f"参数选择拟合期参考数据：{run.n_tuning:,} 行；验证期数据：{run.n_validation:,} 行")
    else:
        print("所有实验均已复用，跳过 CSV 解析、Y 计算和模型拟合。")

    results = current_results()
    for result in results:
        print_experiment_result(result, FEATURE_COLUMNS)

    search_tables = [
        result.search_results.assign(experiment=result.selected_config["name"])
        for result in results if not result.search_results.empty
    ]
    search_summary = pd.concat(search_tables, ignore_index=True) if search_tables else pd.DataFrame()
    export_search_results(search_summary, ALPHA_SEARCH_OUTPUT_PATH)
    if not search_summary.empty:
        print(f"\nalpha 验证明细已导出：{ALPHA_SEARCH_OUTPUT_PATH}")
    export_summary(results_dir, results)
    print(f"实验结果目录：{results_dir}")
    summary = pd.DataFrame([result.metrics for result in results])
    models = {result.selected_config["name"]: result.model for result in results}
    daily_ics = {result.selected_config["name"]: result.daily_ic for result in results}
    print("\n实验汇总：")
    columns = ["name", "alpha", "validation_ic", "n_train", "train_mse", "train_r2", "mean_ic", "std_ic"]
    print(summary[columns].to_string(index=False) if not summary.empty else "没有实验结果。")
    return summary, models, daily_ics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行未完成实验，保存并复用已有结果。")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，重跑当前全部实验配置。")
    parser.add_argument("--results-dir", type=Path, default=None, help="结果目录，默认 analysis/results。")
    args = parser.parse_args()
    summary, models, daily_ics = main(results_dir=args.results_dir, force=args.force)
