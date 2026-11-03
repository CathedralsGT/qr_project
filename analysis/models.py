"""数组接口的线性模型、目标值计算及求解器，不依赖 pandas 或数据路径。"""

import numpy as np


def predict(model, X):
    """所有实验使用相同的线性形式，beta 对应所选特征，b 为截距。"""
    beta, b = model
    return X @ beta + b


def data_loss(y_true, predictions, kind="mse", dates=None):
    """普通损失按行平均；daily 损失先按日平均，再对日期等权。"""
    if kind == "mse":
        return np.mean((y_true - predictions) ** 2)
    if kind in ("mae", "l1"):
        return np.mean(np.abs(y_true - predictions))
    if kind == "lhalf":
        return lhalf_loss(y_true, predictions)
    if kind in ("daily_mse", "daily_centered_mse"):
        y_true = np.asarray(y_true, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        if y_true.ndim != 1 or y_true.shape != predictions.shape:
            raise ValueError("按日损失要求等长的一维标签和预测。")
        residuals = y_true - predictions
        if residuals.ndim != 1 or not np.isfinite(residuals).all():
            raise ValueError("按日损失要求有限的一维标签和预测。")
        groups, counts = _daily_groups(dates, len(residuals))
        if kind == "daily_centered_mse":
            residuals = _daily_demean(residuals, groups, counts)
        return np.mean(np.bincount(groups, weights=residuals ** 2) / counts)
    raise ValueError(f"尚未实现的数据损失：{kind}")


def _daily_groups(dates, n_rows):
    """保留数组行顺序；每日权重只由清洗后实际拟合的日期计数决定。"""
    if dates is None:
        raise ValueError("按日损失和求解器必须提供 dates。")
    dates = np.asarray(dates)
    if dates.ndim != 1 or len(dates) != n_rows or n_rows == 0:
        raise ValueError("dates 必须是与训练数组等长、非空的一维日期数组。")
    # 转换仅用于统一日期表示和识别 NaT，不读取任何未来数据。
    try:
        dates = dates.astype("datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("dates 必须包含有效日期。") from error
    if np.isnat(dates).any():
        raise ValueError("dates 不能包含缺失日期。")
    _, groups, counts = np.unique(dates, return_inverse=True, return_counts=True)
    return groups, counts


def _daily_demean(values, groups, counts):
    """对实际训练截面去均值，支持标签向量和特征矩阵。"""
    if values.ndim == 1:
        means = np.bincount(groups, weights=values) / counts
        return values - means[groups]
    means = np.column_stack([
        np.bincount(groups, weights=values[:, j]) / counts
        for j in range(values.shape[1])
    ])
    return values - means[groups]


def solve_daily_wls(X, y, dates, *, center):
    """求解每日等权 MSE；中心化版本的截距不可识别，固定为零。"""
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("按日求解器要求有限的特征和标签。")
    groups, counts = _daily_groups(dates, len(y))
    sqrt_weight = np.sqrt(1.0 / (len(counts) * counts[groups]))
    if center:
        design = _daily_demean(X, groups, counts)
        target = _daily_demean(y, groups, counts)
        beta = np.linalg.lstsq(design * sqrt_weight[:, None], target * sqrt_weight, rcond=None)[0]
        return beta, 0.0
    design = np.column_stack([X, np.ones(len(y))])
    theta = np.linalg.lstsq(design * sqrt_weight[:, None], y * sqrt_weight, rcond=None)[0]
    return theta[:-1], float(theta[-1])


def penalty(beta, kind="none"):
    """惩罚项只接收 beta，不惩罚截距；强度由外部 alpha 控制。"""
    if kind == "none":
        return 0.0
    if kind == "lasso":
        return np.sum(np.abs(beta))
    if kind == "ridge":
        return np.sum(beta ** 2)
    raise ValueError(f"尚未实现的惩罚项：{kind}")


def objective(model, X, y_train, dates, config):
    """总目标 = 数据损失 + alpha × 惩罚项。"""
    if config["loss"] == "lhalf":
        return lhalf_loss(y_train, predict(model, X), smooth_eps=config.get("smooth_eps", 1e-4)) + config["alpha"] * penalty(model[0], kind=config["penalty"])
    # 仅作为终端输出项，不参与模型拟合
    predictions = predict(model, X)
    loss_value = data_loss(y_train, predictions, kind=config["loss"], dates=dates)
    penalty_value = penalty(model[0], kind=config["penalty"])
    return loss_value + config["alpha"] * penalty_value


def solve_ols(X, y):
    """普通最小二乘求解器，只返回统一的模型表示 (beta, b)。"""

    # 去均值后求普通最小二乘，再还原截距；不使用任何正则项。
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    beta = np.linalg.lstsq(X - X_mean, y - y_mean, rcond=None)[0]
    b = y_mean - X_mean @ beta

    return beta, b


def solve_ridge(X, y, alpha):
    """求解 MSE + alpha * sum(beta ** 2)，截距不参与惩罚。"""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha 必须是非负有限数。")
    if alpha == 0:
        return solve_ols(X, y)

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean

    # 用平均尺度的 Gram 矩阵，使 alpha 不随样本行数变化。
    gram = X_centered.T @ X_centered / len(y)
    rhs = X_centered.T @ y_centered / len(y)
    beta = np.linalg.solve(gram + alpha * np.eye(X.shape[1]), rhs)
    b = y_mean - X_mean @ beta
    return beta, b


def solve_lasso(X, y, alpha, max_iter=10000, tol=1e-10):
    """用坐标下降求解 MSE + alpha * sum(abs(beta))；tol 为最优性条件容差。"""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha 必须是非负有限数。")
    if alpha == 0:
        return solve_ols(X, y)
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter 必须为正整数，tol 必须是正有限数。")

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean

    # 五个特征对应 5×5 矩阵，后续迭代不再扫描全部样本。
    gram = X_centered.T @ X_centered / len(y)
    rhs = X_centered.T @ y_centered / len(y)
    beta = np.zeros(X.shape[1])

    # 本项目用 MSE 而非 MSE/2，所以软阈值为 alpha/2。
    threshold = alpha / 2
    for _ in range(max_iter):
        for j in range(X.shape[1]):
            if gram[j, j] == 0:
                continue  # 常数特征的系数保持为 0。
            partial = rhs[j] - gram[j] @ beta + gram[j, j] * beta[j]
            beta[j] = np.sign(partial) * max(abs(partial) - threshold, 0) / gram[j, j]

        # 最优性检查：非零系数满足 gradient + alpha*sign(beta)=0；
        # 零系数满足 abs(gradient)<=alpha。
        gradient = 2 * (gram @ beta - rhs)
        error = np.maximum(np.abs(gradient) - alpha, 0)
        nonzero = beta != 0
        error[nonzero] = np.abs(gradient[nonzero] + alpha * np.sign(beta[nonzero]))
        if np.max(error) <= tol:
            break
    else:
        raise RuntimeError("Lasso 未在 max_iter 内收敛，请增大 max_iter 或放宽 tol。")

    b = y_mean - X_mean @ beta
    return beta, b


def solve_mae(X, y, alpha=0.0, kind="none", max_iter=10000, tol=1e-7, rho=None):
    """用 ADMM 求解 MAE + alpha * penalty；截距独立求解且不参与惩罚。"""
    if kind not in ("none", "lasso", "ridge"):
        raise ValueError(f"MAE 求解器尚不支持惩罚项：{kind}")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha 必须是非负有限数。")
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter 必须为正整数，tol 必须是正有限数。")

    n, p = X.shape
    A = np.column_stack([X, np.ones(n)])
    theta = np.r_[np.zeros(p), np.median(y)]  # 前 p 项是 beta，最后一项是截距。

    # 只根据训练数据的尺度选迭代步长，不改变损失和 alpha 的定义。
    if rho is None:
        rho = 1 / max(np.median(np.abs(y - np.median(y))), 1e-6)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho 必须是正有限数。")

    # 引入 r=y-A@theta 和 z=beta，分别处理绝对残差与系数惩罚。
    r = y - A @ theta
    z = theta[:p].copy()
    u = np.zeros(n)
    v = np.zeros(p)
    system = A.T @ A / n
    system[:p, :p] += np.eye(p)

    for _ in range(max_iter):
        old_r = r.copy()
        old_z = z.copy()

        # 更新线性模型参数，只需求解 (p+1) 维线性方程。
        rhs = A.T @ (y - r - u) / n
        rhs[:p] += z - v
        theta = np.linalg.solve(system, rhs)
        predictions = A @ theta

        # 更新残差：MAE 对应软阈值操作。
        residual = y - predictions - u
        r = np.sign(residual) * np.maximum(np.abs(residual) - 1 / rho, 0)

        # 更新系数的副本；截距不进入这里。
        coefficients = theta[:p] + v
        if kind == "none":
            z = coefficients
        elif kind == "lasso":
            z = np.sign(coefficients) * np.maximum(np.abs(coefficients) - alpha / rho, 0)
        else:
            z = coefficients / (1 + 2 * alpha / rho)

        residual_error = predictions + r - y
        coefficient_error = theta[:p] - z
        u += residual_error
        v += coefficient_error

        # 同时检查约束误差和参数更新对应的对偶残差。
        primal_error = max(np.sqrt(np.mean(residual_error ** 2)), np.max(np.abs(coefficient_error)))
        dual_change = A.T @ (r - old_r) / n
        dual_change[:p] -= z - old_z
        dual_error = rho * np.max(np.abs(dual_change))
        if max(primal_error, dual_error) <= tol:
            return theta[:p], theta[p]

    raise RuntimeError("MAE 求解器未在 max_iter 内收敛，请增大 max_iter 或调整 tol、rho。")


def lhalf_loss(y_true, predictions, *, smooth_eps=1e-4):
    """实际优化的平滑 L0.5 损失：mean((r² + epsilon²)^(1/4))。"""
    if not np.isfinite(smooth_eps) or smooth_eps <= 0:
        raise ValueError("smooth_eps 必须是正有限数，单位与训练标签相同。")
    residuals = np.asarray(y_true, dtype=float) - np.asarray(predictions, dtype=float)
    return np.mean(np.sqrt(np.hypot(residuals, smooth_eps)))


def _lhalf_irls_start(A, y, epsilon, theta, *, max_iter, tol):
    """固定 epsilon 的 MM/IRLS；仅当归一化参数收敛时返回。"""
    residuals = y - A @ theta
    loss = np.mean(np.sqrt(np.hypot(residuals, epsilon)))
    for _ in range(max_iter):
        radius = np.hypot(residuals, epsilon)
        # w ∝ (r² + epsilon²)^(-3/4)；整体缩放不改变加权最小二乘解。
        weights = (radius / radius.min()) ** -1.5
        gram = A.T @ (weights[:, None] * A)
        rhs = A.T @ (weights * y)
        candidate = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        candidate_residuals = y - A @ candidate
        candidate_loss = np.mean(np.sqrt(np.hypot(candidate_residuals, epsilon)))
        # 理论上 MM 每步不增加目标；仅容许机器舍入量级的误差，否则回溯。
        roundoff = 8 * np.finfo(float).eps * max(1.0, abs(loss))
        for _ in range(30):
            if np.isfinite(candidate_loss) and candidate_loss <= loss + roundoff:
                break
            candidate = (candidate + theta) / 2
            candidate_residuals = y - A @ candidate
            candidate_loss = np.mean(np.sqrt(np.hypot(candidate_residuals, epsilon)))
        else:
            raise RuntimeError("L0.5 IRLS 未能得到下降步，请增大 smooth_eps 或检查输入尺度。")
        step = np.max(np.abs(candidate - theta))
        scale = 1 + np.max(np.abs(theta))
        theta, residuals, loss = candidate, candidate_residuals, candidate_loss
        if step <= tol * scale:
            return theta, loss
    raise RuntimeError("L0.5 IRLS 未在 max_iter 内收敛，请增大 max_iter、smooth_eps 或适当放宽 tol。")


def solve_lhalf(X, y, *, smooth_eps=1e-4, max_iter=1000, tol=1e-6):
    """用 OLS、常数中位数两个确定初值求局部解，按训练平滑损失选优。"""
    if not np.isfinite(smooth_eps) or smooth_eps <= 0:
        raise ValueError("smooth_eps 必须是正有限数，单位与训练标签相同。")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter 必须为正整数。")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol 必须是正有限数。")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("L0.5 求解器要求有限的特征和标签。")
    # 只使用当前拟合样本的统计量改善数值尺度，返回时还原原始线性系数。
    X_mean, X_scale = X.mean(axis=0), X.std(axis=0)
    X_scale = np.where(X_scale > 0, X_scale, 1.0)
    y_mean = y.mean()
    y_scale = max(float(y.std()), smooth_eps)
    A = np.column_stack([(X - X_mean) / X_scale, np.ones(len(y))])
    normalized_y = (y - y_mean) / y_scale
    epsilon = smooth_eps / y_scale
    ols_theta = np.linalg.lstsq(A, normalized_y, rcond=None)[0]
    median_theta = np.r_[np.zeros(X.shape[1]), np.median(normalized_y)]
    candidates = []
    for theta in (ols_theta, median_theta):
        # 任一初值未收敛就报错，不把达到迭代上限伪装为成功。
        candidates.append(_lhalf_irls_start(A, normalized_y, epsilon, theta, max_iter=max_iter, tol=tol))
    theta, _ = min(candidates, key=lambda candidate: candidate[1])
    beta = theta[:-1] * y_scale / X_scale
    return beta, float(y_mean + y_scale * theta[-1] - X_mean @ beta)


def fit(X, y_train, config, *, dates=None):
    """按配置选择匹配的求解器，不允许修改损失后仍默默求 OLS 解。"""
    if "alphas" in config:
        raise ValueError("配置包含多个 alpha，请先通过 select_alpha 选出一个值再拟合。")
    X = np.asarray(X, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if X.ndim != 2 or y_train.ndim != 1 or len(X) != len(y_train) or len(y_train) == 0:
        raise ValueError("X 必须为二维数组，y_train 必须为等长、非空的一维数组。")
    solver = config["solver"]
    if solver == "irls":
        if config["loss"] != "lhalf" or config["penalty"] != "none" or config["alpha"] != 0:
            raise ValueError("当前 irls 只支持 loss='lhalf'、penalty='none'、alpha=0。")
        return solve_lhalf(
            X, y_train, smooth_eps=config.get("smooth_eps", 1e-4),
            max_iter=config.get("max_iter", 1000), tol=config.get("tol", 1e-6),
        )
    if solver == "daily_wls":
        if config["loss"] not in ("daily_mse", "daily_centered_mse"):
            raise ValueError("daily_wls 要求 loss='daily_mse' 或 'daily_centered_mse'。")
        if config["penalty"] != "none" or config["alpha"] != 0:
            raise ValueError("当前 daily_wls 仅支持 penalty='none'、alpha=0。")
        return solve_daily_wls(
            X, y_train, dates, center=config["loss"] == "daily_centered_mse",
        )
    if solver == "admm":
        if config["loss"] not in ("mae", "l1"):
            raise ValueError("当前 ADMM 求解器要求 loss='mae' 或 'l1'。")
        return solve_mae(
            X, y_train, config["alpha"], kind=config["penalty"],
            max_iter=config.get("max_iter", 10000),
            tol=config.get("tol", 1e-7),
            rho=config.get("rho"),
        )

    required_penalty = {"ols": "none", "ridge": "ridge", "lasso": "lasso"}
    if solver not in required_penalty:
        raise ValueError(f"尚未实现的求解器：{solver}")
    if config["loss"] != "mse" or config["penalty"] != required_penalty[solver]:
        raise ValueError(
            f"{solver} 求解器要求 loss='mse'、penalty='{required_penalty[solver]}'。"
        )

    alpha = config["alpha"]

    if solver == "ols":
        if alpha != 0:
            raise ValueError("OLS 要求 alpha=0。")
        return solve_ols(X, y_train)
    if solver == "ridge":
        return solve_ridge(X, y_train, alpha)
    return solve_lasso(
        X, y_train, alpha,
        max_iter=config.get("max_iter", 10000),
        tol=config.get("tol", 1e-10),
    )
