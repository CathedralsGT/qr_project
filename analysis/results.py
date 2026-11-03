"""实验结果的文件保存与复用；不读取行情内容，不参与训练或评价计算。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

if __package__:
    from .config import FEATURE_COLUMNS
    from .result_fingerprint import CORE_MODULES, experiment_code_hash, legacy_pipeline_hash, source_snapshot
else:
    from config import FEATURE_COLUMNS
    from result_fingerprint import CORE_MODULES, experiment_code_hash, legacy_pipeline_hash, source_snapshot


CACHE_VERSION = 2


def _json_value(value):
    """将 NumPy 值和缺失值转换为标准 JSON，保留有限浮点数的精度。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else {"__float__": str(value)}
    if isinstance(value, (Path, datetime, pd.Timestamp)):
        return str(value)
    return value


def _restore_numbers(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {key: _restore_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_numbers(item) for item in value]
    return value


def _json_bytes(value):
    return json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8")


def get_pipeline_hash(source_dir=None):
    """仅供 v1 迁移校验；整模块哈希不再参与新实验编号。"""
    return legacy_pipeline_hash(source_snapshot(source_dir))


def build_context(data_path, split_config, feature_columns):
    """只读取原始文件的元数据和小型代码文件，缓存全命中时无需解析 CSV。"""
    data_path = Path(data_path).resolve()
    info = data_path.stat()
    sources = source_snapshot()
    return _json_value({
        "cache_version": CACHE_VERSION,
        "data": {"path": str(data_path), "size": info.st_size, "mtime_ns": info.st_mtime_ns},
        "split_config": split_config,
        "feature_columns": list(feature_columns),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        # 快照只留在本次运行的内存中；保存时仅写所选实现的哈希。
        "_sources": sources,
        "_legacy_pipeline_hash": legacy_pipeline_hash(sources),
        "_implementation_hashes": {},
    })


def _experiment_context(config, context):
    selected = {key: value for key, value in context.items() if not key.startswith("_") and key != "pipeline_hash"}
    if "feature_columns" in config:
        selected["feature_columns"] = list(config["feature_columns"])
    if "implementation_hash" not in selected:
        config = _json_value(config)
        token = hashlib.sha256(_json_bytes(config)).hexdigest()
        hashes = context.setdefault("_implementation_hashes", {})
        if token not in hashes:
            hashes[token] = experiment_code_hash(config, context["_sources"])
        selected["implementation_hash"] = hashes[token]
    return selected


def experiment_key(config, context):
    """每条声明单独编号；候选 alpha 列表保持输入顺序。"""
    return hashlib.sha256(_json_bytes({"config": config, "context": _experiment_context(config, context)})).hexdigest()


def _experiment_dir(results_dir, config, context):
    name = re.sub(r"[^\w.-]+", "_", str(config["name"])).strip(". ")[:60] or "experiment"
    return Path(results_dir) / f"{name}_{experiment_key(config, context)}"


def _atomic_write(path, content):
    """先写同目录临时文件，再替换目标文件；不留下半个完成标记。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing_", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_result(results_dir, config, context, result):
    """每完成一个实验立即保存；result.json 最后写入，作为完成标记。"""
    context = _experiment_context(config, context)
    folder = _experiment_dir(results_dir, config, context)
    artifacts = {
        "daily_ic.csv": result.daily_ic.to_csv().encode("utf-8-sig"),
        "alpha_search.csv": result.search_results.to_csv(index=False).encode("utf-8-sig"),
    }
    for name, content in artifacts.items():
        _atomic_write(folder / name, content)

    beta, intercept = result.model
    payload = {
        "cache_version": CACHE_VERSION,
        "status": "completed",
        "experiment_id": experiment_key(config, context),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_config": config,
        "context": context,
        "selected_config": result.selected_config,
        "metrics": result.metrics,
        "metric_columns": list(result.metrics),
        "model": {"beta": beta, "intercept": intercept},
        # JSON 为恢复计算对象的依据；CSV 供直接查看，避免 CSV 解析带来类型或精度变化。
        "daily_ic": {
            "dates": [date.isoformat() for date in result.daily_ic.index],
            "values": result.daily_ic.tolist(),
            "name": result.daily_ic.name,
            "index_name": result.daily_ic.index.name,
            "index_dtype": str(result.daily_ic.index.dtype),
            "index_class": type(result.daily_ic.index).__name__,
        },
        "search_results": {
            "columns": result.search_results.columns.tolist(),
            "data": result.search_results.to_numpy().tolist(),
            "dtypes": {column: str(dtype) for column, dtype in result.search_results.dtypes.items()},
        },
        "artifacts": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()},
    }
    _atomic_write(folder / "result.json", _json_bytes(payload))


def _valid_payload(folder):
    try:
        payload = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        expected_key = (
            hashlib.sha256(_json_bytes({"config": payload["input_config"], "context": payload["context"]})).hexdigest()
            if payload["cache_version"] == 1
            else experiment_key(payload["input_config"], payload["context"])
        )
        if (
            payload["status"] != "completed" or payload["cache_version"] not in (1, CACHE_VERSION)
            or payload["experiment_id"] != expected_key
            or set(payload["artifacts"]) != {"daily_ic.csv", "alpha_search.csv"}
        ):
            return None
        for name, expected_hash in payload["artifacts"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected_hash:
                return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None


def save_ic_plot(results_dir, config, context, result, *, overwrite=False):
    """最终评价完成后输出图片；图片独立于数值缓存，缺图时可直接补画。"""
    if __package__:
        from .metrics import plot_daily_ic
    else:
        from metrics import plot_daily_ic

    path = _experiment_dir(results_dir, config, context) / "daily_ic.png"
    if overwrite or not path.is_file():
        plot_daily_ic(result.daily_ic, path, title=config["name"])
    return path


def _migrate_payload(results_dir, old_folder, payload, sources):
    old_context = payload["context"]
    if payload["cache_version"] != 1 or old_context.get("pipeline_hash") != legacy_pipeline_hash(sources):
        return None
    # 保留原数据路径、元数据、NumPy/pandas 版本；不能把 WSL 结果冒充当前 Windows 环境。
    context = dict(old_context, cache_version=CACHE_VERSION, _sources=sources)
    context = _experiment_context(payload["input_config"], context)
    folder = _experiment_dir(results_dir, payload["input_config"], context)
    existing = _valid_payload(folder)
    if existing is not None:
        return existing
    migrated = dict(payload)
    migrated.update(
        cache_version=CACHE_VERSION, context=context,
        experiment_id=experiment_key(payload["input_config"], context),
        legacy_experiment_id=payload["experiment_id"],
        migrated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    for name in payload["artifacts"]:
        _atomic_write(folder / name, (old_folder / name).read_bytes())
    _atomic_write(folder / "result.json", _json_bytes(migrated))
    return migrated


def migrate_legacy_results(results_dir, sources=None):
    """只迁移已验证代码完全一致的 v1 结果，保留旧目录，不训练、不改数值。"""
    sources = source_snapshot() if sources is None else sources
    report = {"migrated": [], "incompatible": [], "invalid": []}
    for path in sorted(Path(results_dir).glob("*/result.json")):
        payload = _valid_payload(path.parent)
        if payload is None:
            report["invalid"].append(path.parent.name)
        elif payload["cache_version"] == 1:
            name = payload["input_config"]["name"]
            result = _migrate_payload(results_dir, path.parent, payload, sources)
            report["migrated" if result is not None else "incompatible"].append(name)
    return report


def load_completed_result(results_dir, config, context, result_type):
    """优先读取 v2；旧记录只有版本和计算口径可证明一致时才能迁移。"""
    key = experiment_key(config, context)
    payload = _valid_payload(_experiment_dir(results_dir, config, context))
    if payload is None and "_sources" in context:
        # 兼容尚未升级、或由仍运行中的 v1 进程刚保存的记录。
        requested = _experiment_context(config, context)
        shared = {k: v for k, v in requested.items() if k not in {"cache_version", "implementation_hash"}}
        for path in sorted(Path(results_dir).glob("*/result.json")):
            old = _valid_payload(path.parent)
            if old is None or old["cache_version"] != 1 or _json_bytes(old["input_config"]) != _json_bytes(config):
                continue
            old_shared = {k: v for k, v in old["context"].items() if k not in {"cache_version", "pipeline_hash"}}
            if old_shared == shared:
                payload = _migrate_payload(results_dir, path.parent, old, context["_sources"])
                if payload is not None:
                    break
    if payload is None or payload["cache_version"] != CACHE_VERSION or payload["experiment_id"] != key:
        return None
    try:

        payload = _restore_numbers(payload)
        daily = payload["daily_ic"]
        if daily["index_class"] == "DatetimeIndex":
            index = pd.DatetimeIndex(pd.to_datetime(daily["dates"]), name=daily["index_name"]).astype(daily["index_dtype"])
        elif daily["index_class"] == "RangeIndex":
            index = pd.RangeIndex(len(daily["values"]), name=daily["index_name"])
        else:
            index = pd.Index(daily["dates"], dtype=daily["index_dtype"], name=daily["index_name"])
        daily_ic = pd.Series(
            daily["values"], dtype=float, name=daily["name"],
            index=index,
        )
        search = payload["search_results"]
        search_results = (
            pd.DataFrame(search["data"], columns=search["columns"]).astype(search["dtypes"])
            if search["columns"] else pd.DataFrame()
        )
        metrics = {
            name: np.nan if payload["metrics"][name] is None else payload["metrics"][name]
            for name in payload["metric_columns"]
        }
        model = np.asarray(payload["model"]["beta"], dtype=float), float(payload["model"]["intercept"])
        return result_type(payload["selected_config"], model, metrics, daily_ic, search_results)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None


def export_summary(results_dir, results):
    """当前配置中已完成的旧、新结果共同汇总；不混入其他历史版本。"""
    rows = []
    for result in results:
        row = dict(result.metrics)
        row.update({
            name: result.selected_config.get(name)
            for name in ("sample_rule", "target", "loss", "penalty", "solver")
        })
        if result.selected_config["loss"] == "lhalf":
            row["smooth_eps"] = result.selected_config.get("smooth_eps", 1e-4)
        row.update({
            "n_features": len(result.model[0]),
            "feature_columns": json.dumps(
                result.selected_config.get("feature_columns", FEATURE_COLUMNS), ensure_ascii=False,
            ),
            "evaluation_pool": result.selected_config.get("evaluation_pool", "common_5f"),
            "train_feature_fill": json.dumps(
                result.selected_config.get("train_feature_fill", {}), ensure_ascii=False,
            ),
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    _atomic_write(Path(results_dir) / "summary.csv", summary.to_csv(index=False).encode("utf-8-sig"))
