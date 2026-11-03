"""比较已构造拟合数据上的候选 alpha，不切分时间或构造训练目标。"""

import numpy as np
import pandas as pd

if __package__:
    from .metrics import evaluate
    from .models import fit
else:
    from metrics import evaluate
    from models import fit


def select_alpha(X_fit, y_fit, validation_data, config, alphas=None, *, feature_columns, dates_fit=None):
    """接收已构造的拟合数组，只比较候选参数；验证始终使用正式 Y。"""
    selected_config = config.copy()
    if alphas is None:
        return selected_config, pd.DataFrame()

    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas 必须是一维、非空的候选列表。")
    if not np.isfinite(alphas).all() or (alphas < 0).any():
        raise ValueError("候选 alpha 必须全部为非负有限数。")
    if X_fit is None or y_fit is None or len(y_fit) == 0 or validation_data is None or validation_data.empty:
        raise ValueError("请提供已构造的非空拟合数组和验证集。")

    results = []
    # 按输入顺序去重；IC 完全相同时选择最先出现的候选。
    for alpha in pd.unique(alphas):
        candidate = selected_config.copy()
        candidate["alpha"] = float(alpha)
        row = {
            "alpha": float(alpha),
            "validation_ic": np.nan,
            "validation_ic_std": np.nan,
            "valid_ic_days": 0,
            "status": "",
        }
        try:
            model = fit(X_fit, y_fit, candidate, dates=dates_fit)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            row["status"] = str(error)
        else:
            _, daily_ic, mean_ic, std_ic = evaluate(model, validation_data, feature_columns)
            row.update({
                "validation_ic": mean_ic,
                "validation_ic_std": std_ic,
                "valid_ic_days": int(daily_ic.notna().sum()),
                "status": "ok" if pd.notna(mean_ic) else "no_valid_ic",
            })
        results.append(row)

    search_results = pd.DataFrame(results)
    if not search_results["validation_ic"].notna().any():
        raise ValueError(
            "所有候选 alpha 均没有有效验证 IC，请检查参数范围或收敛设置。\n"
            + search_results.to_string(index=False)
        )
    best_index = search_results["validation_ic"].idxmax()
    selected_config["alpha"] = float(search_results.loc[best_index, "alpha"])
    return selected_config, search_results
