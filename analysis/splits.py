"""按信号日期划分；越界标签置空，保留计算上下文，不提前清洗。"""


def split_data(data, *, train_start, train_end, test_start, test_end):
    """只按信号日期分组；训练 Y 跨入测试期时置空，行仍保留。"""
    train_data = data.loc[data["日期"].between(train_start, train_end)].copy()
    test_data = data.loc[data["日期"].between(test_start, test_end)].copy()
    train_data["Y"] = train_data["Y"].where(train_data["Y结束日期"] < test_start)
    return train_data, test_data


def validation_split(base_train, validation_start):
    """先按日期切分；内部拟合集在计算 Y_train 前再应用验证截止边界。"""
    fit_data = base_train.loc[base_train["日期"] < validation_start].copy()
    validation_data = base_train.loc[base_train["日期"] >= validation_start].copy()

    if fit_data.empty or validation_data.empty:
        raise ValueError("拟合集或验证集为空，请检查 validation_start 与数据日期范围。")
    return fit_data, validation_data
