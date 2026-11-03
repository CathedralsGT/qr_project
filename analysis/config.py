"""实验声明、输入输出路径与时间窗口；不执行数据处理或拟合。"""

from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "基础数据" / "基础数据.csv"
TRAIN_OUTPUT_PATH = Path(__file__).resolve().parent / "train_data.pkl"
ALPHA_SEARCH_OUTPUT_PATH = Path(__file__).resolve().parent / "alpha_search_results.csv"
FEATURE_COLUMNS = ["原始股价", "成交金额", "总市值", "换手率", "二十日波动率"]
SPLIT_CONFIG = {
    "train_start": "2012-01-01",
    "train_end": "2018-12-31",
    "test_start": "2019-01-01",
    "test_end": "2020-12-31",
    "validation_start": "2018-01-01",
}
# 兼容原常量名；时间窗口的唯一声明位于 SPLIT_CONFIG。
VALIDATION_START = SPLIT_CONFIG["validation_start"]

# 不提供 alphas 时使用固定 alpha；提供 alphas 时，按验证集日均 IC 选择。
# alphas=[0.0] 只评估固定参数的验证 IC，并在完整训练期重新拟合。
# Ridge：penalty="ridge", solver="ridge"；Lasso：penalty="lasso", solver="lasso"。
# 两者的 alpha 均对应 MSE + alpha * penalty 的定义。
# L1 数据损失：loss="mae"（或 "l1"）、solver="admm"，可搭配上述惩罚项。
# feature_columns 仅在需要覆盖默认五特征时填写，列顺序就是系数顺序。
# evaluation_pool="common_5f" 沿用原评价股票池；"selected_features" 仅检查所选输入和正式 Y。
# train_feature_fill 可选：训练输入列到有限常数的映射；省略或 {} 不填补。
EXPERIMENTS = [
    {
        "name": "baseline_ols",
        "alphas": [0.0],
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "ols",
    },
    {
        "name": "ols_5f_fill0",
        "train_feature_fill": {"二十日波动率": 0.0},
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "alphas": [0.0],
        "solver": "ols",
    },
    {
        "name": "ols_5f_fill0_ridge",
        "train_feature_fill": {"二十日波动率": 0.0},
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "ridge",
        "alpha": 0.0,
        "alphas": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0],
        "solver": "ridge",
    },
    {
        "name": "ols_5f_fill0_lasso",
        "train_feature_fill": {"二十日波动率": 0.0},
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "lasso",
        "alpha": 0.0,
        "alphas": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0],
        "solver": "lasso",
    },
    {
        "name": "ols_ridge",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "ridge",
        "alpha": 0.0,
        "alphas": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0],
        "solver": "ridge",
    },
    {
        "name": "ols_lasso",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "lasso",
        "alpha": 0.0,
        "alphas": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "solver": "lasso",
    },
    {
        "name": "mae_no_penalty",
        "alphas": [0.0],
        "sample_rule": "all",
        "target": "raw",
        "loss": "mae",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "admm",
        "max_iter": 20000,
        "tol": 1e-6,
    },
    {
        "name": "mae_no_penalty_fill0",
        "train_feature_fill": {"二十日波动率": 0.0},
        "alphas": [0.0],
        "sample_rule": "all",
        "target": "raw",
        "loss": "mae",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "admm",
        "max_iter": 20000,
        "tol": 1e-6,
    },
    {
        "name": "ols_3f",
        "alphas": [0.0],
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "ols",
    },
    {
        "name": "ols_3f_ridge",
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "ridge",
        "alpha": 0.0,
        "alphas": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0],
        "solver": "ridge",
    },
    {
        "name": "ols_3f_lasso",
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "mse",
        "penalty": "lasso",
        "alpha": 0.0,
        "alphas": [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "solver": "lasso",
    },
    {
        "name": "ols_daily_zscore",
        "alphas": [0.0],
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "ols",
    },
    {
        "name": "lasso_daily_zscore",
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": "mse",
        "penalty": "lasso",
        "alpha": 0.0,
        "alphas": [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        "solver": "lasso",
    },
    {
        "name": "ols_3f_zscore",
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "solver": "ols",
    },
    {
        "name": "ols_3f_same_rows",
        "feature_columns": ["原始股价", "成交金额", "总市值"],
        "evaluation_pool": "common_5f",
        "sample_rule": "complete_5f",
        "target": "raw",
        "loss": "mse",
        "penalty": "none",
        "alpha": 0.0,
        "alphas": [0.0],
        "solver": "ols",
    },
    {
        "name": "ols_zscore_daily_equal",
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": "daily_mse",
        "penalty": "none",
        "alpha": 0.0,
        "alphas": [0.0],
        "solver": "daily_wls",
    },
    {
        "name": "ols_zscore_daily_centered",
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "daily_zscore",
        "loss": "daily_centered_mse",
        "penalty": "none",
        "alpha": 0.0,
        "alphas": [0.0],
        "solver": "daily_wls",
    },
    {
        "name": "lhalf_no_penalty",
        "evaluation_pool": "common_5f",
        "sample_rule": "all",
        "target": "raw",
        "loss": "lhalf",
        "penalty": "none",
        "alpha": 0.0,
        "alphas": [0.0],
        "solver": "irls",
        "smooth_eps": 1e-4,
        "max_iter": 1000,
        "tol": 1e-6,
    },
]


def get_training_config(experiment):
    """向训练数据层传递样本、目标和可选的输入填补规则。"""
    config = {key: experiment[key] for key in ("sample_rule", "target")}
    if "train_feature_fill" in experiment:
        config["train_feature_fill"] = experiment["train_feature_fill"]
    return config


def get_model_config(experiment):
    """模型只接收单次拟合参数；候选 alphas 由编排层单独交给搜索层。"""
    config = {key: experiment[key] for key in ("loss", "penalty", "alpha", "solver")}
    config.update({key: experiment[key] for key in ("max_iter", "tol", "rho") if key in experiment})
    if experiment["solver"] == "irls":
        config.update({key: experiment[key] for key in ("smooth_eps",) if key in experiment})
    return config
