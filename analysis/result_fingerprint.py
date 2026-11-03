"""按配置裁剪字面量分派并跟踪直接函数依赖；不执行或改写计算代码。"""

import ast
import copy
import hashlib
import json
from pathlib import Path


CORE_MODULES = ("data.py", "splits.py", "training_data.py", "models.py", "metrics.py", "selection.py")
UNKNOWN = object()


def source_snapshot(source_dir=None):
    folder = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    return {
        path.name: path.read_text(encoding="utf-8-sig")
        for path in folder.glob("*.py")
        if path.name not in {"results.py", "result_fingerprint.py", "__init__.py"}
    }


def _digest(value):
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def legacy_pipeline_hash(sources):
    """原 v1 算法，仅用于证明旧记录可迁移，不参与新实验编号。"""
    code = {
        name: ast.dump(ast.parse(sources[name]), include_attributes=False)
        for name in CORE_MODULES
    }
    for name, functions in (
        ("config.py", {"get_training_config", "get_model_config"}),
        ("analysis.py", {"run_experiment", "run_experiments"}),
    ):
        code[name] = [
            ast.dump(node, include_attributes=False) for node in ast.parse(sources[name]).body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
    return _digest(code)


def _value(node, known):
    """仅解释字面量、已知选择器和简单比较；不使用 eval，不执行函数。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return known.get(node.id, UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_value(item, known) for item in node.elts]
        if any(item is UNKNOWN for item in values):
            return UNKNOWN
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        keys, values = [_value(item, known) for item in node.keys], [_value(item, known) for item in node.values]
        if any(item is UNKNOWN for item in keys + values):
            return UNKNOWN
        return dict(zip(keys, values))
    if isinstance(node, ast.Subscript):
        container, key = _value(node.value, known), _value(node.slice, known)
        if isinstance(container, (dict, list, tuple)) and key is not UNKNOWN:
            try:
                return container[key]
            except (KeyError, IndexError, TypeError):
                return UNKNOWN
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        container = _value(node.func.value, known)
        args = [_value(arg, known) for arg in node.args]
        if isinstance(container, dict) and 1 <= len(args) <= 2 and not any(arg is UNKNOWN for arg in args):
            return container.get(*args)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _value(node.left, known), _value(node.comparators[0], known)
        if left is not UNKNOWN and right is not UNKNOWN:
            try:
                op = node.ops[0]
                if isinstance(op, ast.Eq): return left == right
                if isinstance(op, ast.NotEq): return left != right
                if isinstance(op, ast.In): return left in right
                if isinstance(op, ast.NotIn): return left not in right
                if isinstance(op, (ast.Is, ast.IsNot)) and (right is None or isinstance(right, bool)):
                    return (left is right) if isinstance(op, ast.Is) else (left is not right)
            except TypeError:
                pass
    if isinstance(node, ast.BoolOp):
        values = [_value(item, known) for item in node.values]
        if all(isinstance(item, bool) for item in values):
            return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _value(node.operand, known)
        if isinstance(value, bool): return not value
    return UNKNOWN


def _literal(value):
    """已知映射可能含 UNKNOWN，不把这种部分已知的对象替换为字面量。"""
    try:
        return ast.parse(repr(value), mode="eval").body if value is not UNKNOWN else None
    except SyntaxError:
        return None


class _Specialize(ast.NodeTransformer):
    def __init__(self, known):
        self.known = dict(known)

    def block(self, body):
        output = []
        for statement in body:
            result = self.visit(statement)
            output.extend(result if isinstance(result, list) else [result])
            if output and isinstance(output[-1], (ast.Return, ast.Raise)):
                break
        return output

    def forget_writes(self, nodes):
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    self.known.pop(child.id, None)
                if isinstance(child, (ast.Subscript, ast.Attribute)) and isinstance(child.ctx, ast.Store):
                    root = child.value
                    while isinstance(root, (ast.Subscript, ast.Attribute)): root = root.value
                    if isinstance(root, ast.Name): self.known.pop(root.id, None)

    def visit_If(self, node):
        condition = _value(node.test, self.known)
        if isinstance(condition, bool):
            return self.block(node.body if condition else node.orelse)
        # 未知条件保留两条路径；分支内的赋值不传播到另一条路径。
        self.forget_writes([node.test])
        before = dict(self.known)
        written = copy.deepcopy([node.test] + node.body + node.orelse)
        node.test = self.visit(node.test)
        node.body = self.block(node.body)
        self.known = dict(before)
        node.orelse = self.block(node.orelse)
        self.known = before
        self.forget_writes(written)
        return node

    def visit_Assign(self, node):
        value = _value(node.value, self.known)
        node.value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.known[target.id] = value
            else:
                self.forget_writes([target])
        return node

    def visit_AnnAssign(self, node):
        value = _value(node.value, self.known)
        node = self.generic_visit(node)
        self.forget_writes([node])
        if isinstance(node.target, ast.Name): self.known[node.target.id] = value
        return node

    def visit_AugAssign(self, node):
        self.forget_writes([node])
        return self.generic_visit(node)

    def visit_NamedExpr(self, node):
        value = _value(node.value, self.known)
        node.value = self.visit(node.value)
        self.forget_writes([node.target])
        if isinstance(node.target, ast.Name): self.known[node.target.id] = value
        return node

    def _conservative(self, node):
        self.forget_writes([node])
        return node

    visit_Try = visit_TryStar = visit_With = visit_AsyncWith = visit_Match = _conservative

    def visit_For(self, node):
        before = dict(self.known)
        written = copy.deepcopy([node.target] + node.body + node.orelse)
        self.forget_writes(written)
        node.iter = self.visit(node.iter)
        node.body = self.block(node.body)
        self.known = dict(before)
        self.forget_writes(written)
        node.orelse = self.block(node.orelse)
        self.known = before
        self.forget_writes(written)
        return node

    def visit_While(self, node):
        before = dict(self.known)
        written = copy.deepcopy(node.body + node.orelse)
        self.forget_writes(written)
        node.test = self.visit(node.test)
        node.body = self.block(node.body)
        self.known = dict(before)
        self.forget_writes(written)
        node.orelse = self.block(node.orelse)
        self.known = before
        self.forget_writes(written)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            literal = _literal(self.known.get(node.id, UNKNOWN))
            if literal is not None: return literal
        return node

    def visit_Subscript(self, node):
        literal = _literal(_value(node, self.known))
        return literal if literal is not None else self.generic_visit(node)


def _pruned(function, known):
    function = copy.deepcopy(function)
    function.body = _Specialize(known).block(function.body)
    # 常量折叠后可能不再读取 required_penalty 等字面量表；去除这类无副作用的死赋值。
    reads = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    def trim(body):
        for statement in body:
            if isinstance(statement, ast.Assign) and all(isinstance(t, ast.Name) and t.id not in reads for t in statement.targets):
                try:
                    ast.literal_eval(statement.value)
                    continue
                except (ValueError, TypeError):
                    pass
            for field in ("body", "orelse", "finalbody"):
                if hasattr(statement, field): setattr(statement, field, list(trim(getattr(statement, field))))
            yield statement
    function.body = list(trim(function.body))
    return function


def experiment_code_hash(config, sources):
    """所选分支、可达直接函数及使用的全局常量参与哈希；未知分派保守保留。"""
    config = dict(sorted(config.items()))
    trees = {name: ast.parse(source) for name, source in sources.items()}
    functions = {name: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for name, tree in trees.items()}
    imports, globals_ = {}, {}
    for name, tree in trees.items():
        imports[name], globals_[name] = {}, {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[name][alias.asname or alias.name] = (f"{node.module}.py", alias.name)
                    globals_[name][alias.asname or alias.name] = ast.dump(
                        ast.ImportFrom(module=node.module, names=[alias], level=node.level), include_attributes=False,
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    globals_[name][alias.asname or alias.name] = ast.dump(ast.Import(names=[alias]), include_attributes=False)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name): globals_[name][target.id] = ast.dump(node, include_attributes=False)

    model_config = {key: config[key] for key in ("solver", "loss", "penalty")}
    model_config.update({key: config[key] for key in ("max_iter", "tol", "rho", "smooth_eps") if key in config})
    model_config["alpha"] = UNKNOWN  # 候选 alpha 与最终 alpha 可不同，不能按声明中的占位值裁剪。
    training_config = {key: config[key] for key in ("sample_rule", "target")}
    if "train_feature_fill" in config:
        training_config["train_feature_fill"] = config["train_feature_fill"]
    signatures, visited = {}, set()

    def collect(module, name, supplied=None):
        function = functions.get(module, {}).get(name)
        if function is None: return
        known = {"experiment": config, "config": config}
        if module in {"models.py", "metrics.py", "selection.py"}: known["config"] = model_config
        if module == "training_data.py": known["config"] = training_config
        if name == "select_alpha": known["alphas"] = config.get("alphas")
        for arg, default in zip(function.args.args[-len(function.args.defaults):], function.args.defaults) if function.args.defaults else []:
            value = _value(default, {})
            if value is not UNKNOWN: known[arg.arg] = value
        if supplied: known.update(supplied)
        token = (module, name, repr(known))
        if token in visited: return
        visited.add(token)
        active = _pruned(function, known)
        signature = f"{module}:{name}:{len(signatures)}"
        signatures[signature] = ast.dump(active, include_attributes=False)
        locals_ = {arg.arg for arg in active.args.args + active.args.kwonlyargs}
        locals_.update(n.id for n in ast.walk(active) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        for node in ast.walk(active):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in locals_:
                if node.id in globals_[module]: signatures[f"{module}:global:{node.id}"] = globals_[module][node.id]
                if node.id in imports[module]:
                    target_module, target_name = imports[module][node.id]
                    if target_module in globals_ and target_name in globals_[target_module]:
                        signatures[f"{target_module}:global:{target_name}"] = globals_[target_module][target_name]
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name): continue
            target_module, target_name = (module, node.func.id)
            if target_name not in functions[module]: target_module, target_name = imports[module].get(target_name, (module, target_name))
            callee = functions.get(target_module, {}).get(target_name)
            if callee is None: continue
            bindings = {}
            for arg, value in zip(callee.args.args, node.args):
                literal = _value(value, {})
                if literal is not UNKNOWN or arg.arg not in {"config", "experiment"}: bindings[arg.arg] = literal
            for keyword in node.keywords:
                literal = _value(keyword.value, {})
                if keyword.arg is not None and (literal is not UNKNOWN or keyword.arg not in {"config", "experiment"}): bindings[keyword.arg] = literal
            collect(target_module, target_name, bindings)

    collect("data.py", "prepare_data")
    collect("splits.py", "split_data")
    if config.get("alphas") is not None: collect("splits.py", "validation_split")
    collect("analysis.py", "run_experiment")
    return _digest(signatures)
