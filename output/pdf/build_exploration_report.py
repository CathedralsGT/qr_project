"""Build the exploration report from saved results; do not refit or change experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "exploration_report_assets"
ASSETS.mkdir(exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted
from pypdf import PdfReader

SUMMARY = ROOT / "analysis/results/summary.csv"
df = pd.read_csv(SUMMARY).set_index("name")
shutil.copy2(SUMMARY, ASSETS / "summary_snapshot.csv")
records, series, manifest = {}, {}, {}
for name, row in df.iterrows():
    candidates = []
    for path in (ROOT / "analysis/results").glob(name + "_*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["input_config"]["name"] != name:
            continue
        if all(np.isclose(record["metrics"][key], row[key], rtol=0, atol=1e-12) for key in ("mean_ic", "std_ic", "n_train", "n_eval", "alpha")):
            candidates.append((record["completed_at_utc"], path, record))
    if not candidates:
        raise ValueError(f"No matching result for {name}")
    _, path, record = max(candidates, key=lambda x: x[0])
    records[name] = record
    series[name] = pd.Series(record["daily_ic"]["values"], index=pd.to_datetime(record["daily_ic"]["dates"])).dropna()
    assert len(series[name]) == int(row["valid_ic_days"])
    shutil.copy2(path, ASSETS / f"{name}.json")
    png = path.parent / "daily_ic.png"
    if png.exists():
        shutil.copy2(png, ASSETS / f"{name}.png")
    manifest[name] = {"result": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "png": str(png.relative_to(ROOT)) if png.exists() else None}

def hac(s, lag):
    x = s.to_numpy(); n = len(x); c = x - x.mean()
    var = c @ c / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (c[k:] @ c[:-k]) / n
    se = np.sqrt(var / n)
    return {"lag": lag, "se": float(se), "t": float(x.mean()/se), "ci95": [float(x.mean()-1.96*se), float(x.mean()+1.96*se)]}

stats = {}
for name, s in series.items():
    stats[name] = {"annual": {str(y): {"mean": float(g.mean()), "std": float(g.std()), "n": len(g)} for y,g in s.groupby(s.index.year)},
                   "positive_fraction": float((s>0).mean()), "acf1": float(s.autocorr(1)), "hac5": hac(s,5), "hac20": hac(s,20)}

# Independently audit the added training rows, using the same target boundary.
audit_path = ASSETS / "added_rows_audit.json"
if not audit_path.exists():
    from analysis import config, data, splits, training_data
    panel = data.prepare_data(config.DATA_PATH)
    train, test = splits.split_data(panel, **{k:v for k,v in config.SPLIT_CONFIG.items() if k != "validation_start"})
    del panel, test
    three = training_data.prepare_training_data(train, {"target":"raw", "sample_rule":"all"}, cutoff="2019-01-01", feature_columns=config.FEATURE_COLUMNS[:3])
    mask = training_data.make_selection_flag(three, "complete_5f", cutoff="2019-01-01")
    extra, complete = three.loc[~mask], three.loc[mask]
    audit = {"added_rows": len(extra), "added_dates": int(extra["日期"].nunique()), "all_fit_dates": int(three["日期"].nunique()),
             "added_stocks": int(extra["股票代码"].nunique()), "by_year": {str(k):int(v) for k,v in extra.groupby(extra["日期"].dt.year).size().items()},
             "missing_volatility": int(extra["二十日波动率"].isna().sum()), "missing_turnover": int(extra["换手率"].isna().sum()),
             "complete_y_mean": float(complete.Y.mean()), "extra_y_mean": float(extra.Y.mean()),
             "complete_y_std": float(complete.Y.std()), "extra_y_std": float(extra.Y.std()),
             "data_size": config.DATA_PATH.stat().st_size, "data_mtime_ns": config.DATA_PATH.stat().st_mtime_ns}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    del train, three, extra, complete
else:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert audit["added_rows"] == 285576
(ASSETS / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
(ASSETS / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# A derived comparison uses the saved daily series, with labels in Chinese.
fontpath = "C:/Windows/Fonts/simhei.ttf"
fp = FontProperties(fname=fontpath)
plt.rcParams.update({"font.family": fp.get_name(), "axes.unicode_minus": False, "font.size": 10})
names = ["baseline_ols", "ols_5f_fill0", "mae_no_penalty", "mae_no_penalty_fill0"]
labels = ["删行 OLS", "填补 OLS", "删行 MAE", "填补 MAE"]
fig, ax = plt.subplots(figsize=(8.6,3.3), dpi=200)
x = np.arange(4)
for i, (year, color) in enumerate([("2019", "#123b50"), ("2020", "#0a8b85")]):
    values = [stats[n]["annual"][year]["mean"] for n in names]
    bars = ax.bar(x+(i-.5)*.32, values, width=.32, color=color, label=year)
    ax.bar_label(bars, labels=[f"{v:.5f}" for v in values], fontsize=9, padding=4)
ax.set_xticks(x, labels); ax.set_ylim(0,.032); ax.set_ylabel("年平均 Pearson IC")
ax.legend(frameon=False, ncol=2, loc="upper right"); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
fig.tight_layout(pad=1.2); fig.savefig(ASSETS/"annual_comparison.png", bbox_inches="tight"); plt.close(fig)

pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CNH", fontpath))
pdfmetrics.registerFontFamily("CN", normal="CN", bold="CNH", italic="CN", boldItalic="CNH")
NAVY = colors.HexColor("#123b50"); TEAL = colors.HexColor("#087f7b"); INK = colors.HexColor("#243845")
MUTED = colors.HexColor("#607382"); LIGHT = colors.HexColor("#eef5f6"); RULE = colors.HexColor("#d8e2e7")
W,H = A4; WIDTH = W-96
styles = {
    "body": ParagraphStyle("body", fontName="CN", fontSize=10.3, leading=16.2, textColor=INK, spaceAfter=7, wordWrap="CJK"),
    "small": ParagraphStyle("small", fontName="CN", fontSize=8.8, leading=13, textColor=MUTED, spaceAfter=5, wordWrap="CJK"),
    "h1": ParagraphStyle("h1", fontName="CNH", fontSize=20, leading=27, textColor=NAVY, spaceAfter=12, wordWrap="CJK"),
    "h2": ParagraphStyle("h2", fontName="CNH", fontSize=12, leading=18, textColor=TEAL, spaceBefore=8, spaceAfter=6, wordWrap="CJK"),
    "kicker": ParagraphStyle("kicker", fontName="CNH", fontSize=8.5, leading=13, textColor=TEAL, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="CN", fontSize=9.0, leading=13, textColor=INK, wordWrap="CJK"),
    "th": ParagraphStyle("th", fontName="CNH", fontSize=9.0, leading=13, textColor=colors.white, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.0, leading=11, textColor=INK),
}
story=[]; editable=[]
def p(text, style="body"):
    story.append(Paragraph(text, styles[style])); editable.append(text)
def title(num, text, sub=None):
    if story: story.append(PageBreak())
    p(f"研究记录  /  {num:02d}", "kicker"); p(text,"h1")
    if sub: p(sub,"small")
def h(text): p(text,"h2")
def callout(text):
    t=Table([[Paragraph(text,styles["body"])]],colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),.5,RULE),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,9)]); editable.append(text)
def table(headers, rows, widths, small=False):
    cs = styles["cell"] if not small else ParagraphStyle("tiny", parent=styles["cell"], fontSize=8.2, leading=11.4)
    content=[[Paragraph(escape(str(x)),styles["th"]) for x in headers]]+[[Paragraph(escape(str(x)),cs) for x in row] for row in rows]
    t=Table(content,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7f8")]),("LINEBELOW",(0,0),(-1,0),.6,NAVY),("LINEBELOW",(0,-1),(-1,-1),.5,RULE),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,8)]); editable.append(" | ".join(headers)); editable.extend(" | ".join(map(str,row)) for row in rows)
def picture(name, caption, width=WIDTH):
    path=ASSETS/name
    from PIL import Image as PILImage
    with PILImage.open(path) as img: iw,ih=img.size
    story.append(Image(str(path),width=width,height=width*ih/iw)); p(caption,"small")
def metric_rows(items):
    return [[label,f"{df.loc[n,'validation_ic']:.6f}",f"{df.loc[n,'mean_ic']:.6f}",f"{df.loc[n,'std_ic']:.6f}"] for n,label in items]

title(1,"五因子线性模型：实验探索与初步结论","JHSS_QUANT 编程课题  |  阶段性研究报告  |  2026 年 11 月 1 日")
callout("<b>核心结论：</b>当前最大的一项数值改善来自保留波动率缺失的训练样本。五特征 OLS 的测试平均 IC 从 0.012861 提高至 0.018384；同样本消融并不支持“三特征本身更优”的最初猜想。若将填补 OLS 视为主基线，目前尚未获得明确优于它的组合方法。")
h("1. 研究问题与评价口径")
p("题目限定使用五个给定输入，保持普通多元线性预测结构，通过损失函数、训练标签或训练样本选取改进日均截面 IC。输入已做 rank zscore；本文不新增外部数据。三特征模型只用于诊断性消融，不作为最终五特征方案。")
table(["环节","统一约定"],[
    ["正式标签","Y = C(t+5) / O(t+1) / 5；遵循题目原式，不减 1。"],
    ["时间切分","2012-2018 年训练；2018 年作内部验证；选参后在完整训练期重训；2019-2020 年评价。"],
    ["样本边界","标签按市场交易日键连接；内部拟合与最终重训分别剔除跨越各自截止边界的标签。"],
    ["共同评价池","所有表中实验均使用 common_5f：测试共 1,366,724 行、482 个有效交易日。只改变训练处理，不扩大评价股票池。"],
    ["指标","每天计算预测值与正式 Y 的 Pearson 相关，再对日期等权平均；std_IC 为逐日 IC 的样本标准差。"],
], [81,WIDTH-81])
h("2. 探索路径")
p("复现删行 OLS → 加入 Ridge / Lasso → 根据 Lasso 零系数提出三特征假设 → 三特征结果改善 → 发现训练样本不一致 → 同样本对照推翻原归因 → 填补波动率并保留五特征 → 检验正则化、MAE 与填补能否叠加。标签标准化和按日损失作为并行分支进行比较。")
h("3. 结论边界")
p("题目没有规定缺失值处理方式。因此，本文将最初基准明确称为“删行 OLS”，将“训练填补 OLS”作为更严格的比较基线。填补只发生在训练期；如果外部基准连验证、测试也填补，其数值不能与本文直接比较。当前评价期已被反复查看，结果属于探索性证据，而非独立的最终泛化检验。")

title(2,"让每个假设对应一项可复现配置","工程设计服务于控制变量：计算流程共享，实验差异显式声明。")
h("1. 从单一脚本到分层实验流程")
table(["模块","作用"],[
 ["config.py / analysis.py","声明实验、时间窗口与输入列；编排内部验证、最终重训、评价和输出。"],
 ["data.py / splits.py","在完整行情上构造正式标签；按日期切分并屏蔽跨界标签，不提前丢掉计算上下文。"],
 ["training_data.py","在当前拟合期生成 Y_train、筛选标志，随后填补指定输入并清洗。"],
 ["models.py / selection.py","数组接口求解线性系数；在固定验证口径下选择 alpha。模型不负责改变股票池。"],
 ["metrics.py / results.py","计算训练指标和正式 IC；保存配置、系数、搜索表、逐日 IC 和图片，支持结果复用。"],
], [145,WIDTH-145])
h("2. config 的扩展由实际对照问题驱动")
table(["需要回答的问题","配置演进"],[
 ["换正则项或损失，究竟改变什么？","分离 loss、penalty、alpha、solver；只允许匹配组合，惩罚不作用于截距。"],
 ["固定模型怎样获得同一验证期的分数？","使用 alphas=[0.0]；多候选按验证 IC 选 alpha，再完整重训。"],
 ["减少特征还是扩大样本带来提升？","加入 feature_columns 与 evaluation_pool，再用 sample_rule=complete_5f 固定训练行。"],
 ["标签标准化是否等于日期等权？","分离 target=daily_zscore 与 daily_mse / daily_centered_mse，后两者用 daily_wls。"],
 ["填补能否独立检验？","新增 train_feature_fill；省略时行为不变，显式配置时只填补训练输入。"],
], [157,WIDTH-157])
h("3. 缺失填补实验的声明")
code='''{
    "name": "ols_5f_fill0", "sample_rule": "all",
    "target": "raw", "loss": "mse", "penalty": "none",
    "alpha": 0.0, "alphas": [0.0], "solver": "ols",
    "evaluation_pool": "common_5f",
    "train_feature_fill": {"二十日波动率": 0.0}
}'''
# Render code with a Chinese-capable font to preserve the actual column name.
story.append(Preformatted(code,ParagraphStyle("codeCN",parent=styles["code"],fontName="CN",fontSize=9.1,leading=12.5))); editable.append(code)
p("执行顺序：日期边界 → 训练标签 → 原始特征上的样本筛选 → 非有限值转 NaN → 指定输入填补 → 剩余缺失清洗 → 拟合。complete_5f 会先删缺失行，所以扩样本实验必须使用 all。", "small")
p("扩展填补功能时，93 项测试通过；14 个已有实验在同一合成面板上的系数、指标、逐日 IC 与选参结果逐项完全一致。此检查验证软件行为一致性，不等于证明统计方法有效。", "small")

title(3,"一次被对照实验推翻的特征解释","假设 H1：Lasso 压零的两个特征没有贡献，删去它们应当改善预测。")
p("最初删行 OLS 的测试平均 IC 为 0.012861。Ridge 几乎没有改变结果；Lasso 选取 alpha=1e-4 后，测试 IC 为 0.014484，且将换手率和二十日波动率系数压到 0。由此提出诊断性猜想：只用原始股价、成交金额和总市值，是否能得到更好的 OLS？")
table(["实验","训练行数","验证 IC","测试 IC"],[
 ["五特征删行 OLS","3,352,997","0.041747","0.012861"],
 ["三特征 OLS（自然清洗）","3,638,573","0.042243","0.015740"],
 ["三特征 OLS（complete_5f）","3,352,997","0.041071","0.012845"],
], [183,100,108,WIDTH-391])
h("第一次观察：结果改善，但同时改变了两个变量")
p("三特征 OLS 相对删行基准的平均 IC 增加约 0.002879。然而训练数据多了 285,576 行：不再使用波动率后，波动率缺失不再触发训练删行。此时“特征减少”与“样本扩大”混在一起，不能将改善直接归因于特征选择。")
h("控制变量：固定训练行后，三特征优势消失")
p("sample_rule=complete_5f 先要求原五特征全部有限，再把前三列交给模型。这个版本的测试 IC 为 0.012845，与五特征基准几乎相同且略低；验证 IC 也下降。因此，当前对照不支持“三特征在相同样本上更优”。")
callout("<b>假设修正：</b>三特征自然清洗版本的改善，主要沿着“保留原本被删掉的训练行”这条路径出现。Lasso 的零系数是特定样本、损失和正则强度下的选择结果，不是特征普遍无预测信息的证明。")
h("后续反证：扩样本后，Lasso 保留的变量发生变化")
table(["训练处理 / Lasso","原始股价","换手率","波动率"],[
 ["删缺失行，alpha=1e-4","-0.000016","0","0"],
 ["波动率填 0，alpha=1e-4","0","-0.000235","0.000136"],
], [191,102,102,WIDTH-395])
p("填补后的 Lasso 保留了原先被压零的两个变量，反而压零原始股价系数。这进一步表明，特征作用需要放在具体训练分布与其他输入的条件下讨论；不能据第一次稀疏解直接删除特征。")

title(4,"保留缺失样本：当前最大的数值改善","假设 H2：恢复被删除的训练记录，同时保留五特征，能够改善模型。")
p("将训练波动率缺失值填为 0，保留其他四个已观测输入和有效标签；它是标准化输入尺度上的常数填补，不是重新算出真实波动率。训练行数增至 3,638,573，测试平均 IC 为 0.018384，std_IC 从 0.115780 降至 0.109039。共同评价池保持不变。")
picture("baseline_ols.png","图 1  原始 results 图：删行 OLS。浅蓝为逐日 IC，橙色为后向 20 个观测日均值，虚线为年均值。")
picture("ols_5f_fill0.png","图 2  原始 results 图：训练填补 OLS。平均表现提高，但图中仍存在持续的负 IC 区间。两图纵轴范围以原图刻度为准。")
p(f"新增记录并非集中在“几天”：共 {audit['added_rows']:,} 行、{audit['added_stocks']:,} 只股票，覆盖最终训练期全部 {audit['added_dates']:,} 个交易日，缺失均来自波动率。新增样本 Y 的标准差为 {audit['extra_y_std']:.5f}，原完整样本为 {audit['complete_y_std']:.5f}。", "small")
p("解释边界：扩样本同时改变了数量与分布。现有结果不能证明收益只来自更多数据，也不能证明 0 是最优填充值；尚未识别波动率缺失的原始成因。", "small")

title(5,"组合未形成简单叠加：正则化与 MAE","假设 H3：填补与其他单项改进组合后，能进一步提高 IC。")
table(["五特征实验","验证 IC","测试 IC","std_IC"], metric_rows([
 ("baseline_ols","删行 OLS"),("mae_no_penalty","删行 MAE"),("ols_5f_fill0","填补 OLS"),
 ("mae_no_penalty_fill0","填补 MAE"),("ols_5f_fill0_ridge","填补 Ridge"),("ols_5f_fill0_lasso","填补 Lasso")]),[175,108,108,WIDTH-391])
h("1. MAE 的增益依赖训练样本处理")
p("删行时，将 MSE 换为 MAE，测试 IC 增加 0.003993；填补后再将 MSE 换为 MAE，测试 IC 反而减少 0.000221，std_IC 约提高 14.2%，验证 IC 也更低。两项操作的测试收益没有简单叠加。均值差异很小，不能据此宣称 MAE 显著劣于 OLS。")
p("一种待检验的机制是：新增样本改变了特征与标签的关系，而 MAE 对大残差的相对重视程度低于 MSE，因此从新增样本获得的收益较少。标签波动较大不等于残差更大；目前没有残差分组或影响度分析来确认这一机制。")
h("2. 正则化的小幅验证优势没有转为测试优势")
p("填补 Ridge 选 alpha=0.1，填补 Lasso 选 alpha=1e-4，两者验证 IC 略高于填补 OLS，但测试 IC 均更低。结论只针对已搜索网格：填补 Lasso 从 1e-4 起步，缺少 0 与更弱惩罚候选，较强候选大量出现 no_valid_ic，不能据此否定所有正则强度。")
picture("mae_no_penalty_fill0.png","图 3  原始 results 图：填补 MAE。与图 2 对照，均值接近并不意味着逐日路径或波动相同。")

title(6,"并行分支与完整实验结果","结果快照共 18 项；所有测试指标来自同一 common_5f 评价口径。")
p("日标准化只改变 Y_train；按日等权与截面中心化是另外两个操作。它们在原完整训练样本上主要降低 std_IC，没有形成较大的均值增益。平滑残差 L0.5 也未超过 MAE，更未超过填补 OLS。以下列出全部结果，避免只展示有利实验。")
rows=[]
for name,row in df.iterrows():
    rows.append([name, "扩充" if int(row.n_train)==3638573 else "完整", "-" if pd.isna(row.validation_ic) else f"{row.validation_ic:.6f}",f"{row.mean_ic:.6f}",f"{row.std_ic:.6f}"])
table(["实验名称","训练集","验证 IC","测试 IC","std_IC"],rows,[224,42,78,78,WIDTH-422],small=True)
p("“完整”指五特征完整样本，共 3,352,997 行；“扩充”指 3,638,573 行。三特征自然清洗与五特征填补使用的输入数不同。ols_3f_zscore 未声明 alphas，故未计算内部验证 IC。", "small")
p("L0.5 实际优化 mean((r²+epsilon²)^(1/4))，epsilon=1e-4，使用双初值 IRLS 求局部解；不是系数 L0.5 正则。它的训练 R² 为负并不与正 IC 矛盾：训练损失与测试截面相关并非同一个目标。", "small")
p("这些分支尚未全部在统一填补基线上重新检验。daily_zscore 也不等同于题目提示的截面 rank 标签变换；本阶段未完成 rank 标签及低波动样本筛选实验。", "small")

title(7,"平均 IC 为正，仍不足以确认稳定预测能力","补充诊断基于保存的 daily_ic 数据；不重新拟合，也不改变选参结果。")
picture("annual_comparison.png","图 4  根据保存的逐日 IC 重绘年均值。2019 年 244 天、2020 年 238 天；各年均值按当年有效日期等权计算。")
p("填补 OLS 在两年均优于删行 OLS。填补 MAE 相比填补 OLS，在 2019 年略低、2020 年略高：其 2020 年均 IC 为 0.012273，不能把图中的某段滚动低值当作全年均值。即使改进相对基准有效，2020 年的绝对信号仍较弱。")
h("考虑重叠收益窗口与逐日相关性")
p("填补 OLS 的 IC 均值为 0.018384，正 IC 日期占 57.1%，一阶自相关约 0.815。未来收益窗口重叠，加上特征和行情持续性，使 482 天不能视为 482 次独立试验。用日均 IC 序列估计均值的标准误，而不是把百万股票行数当作独立样本数。")
stat_names=[("baseline_ols","删行 OLS"),("ols_5f_fill0","填补 OLS"),("mae_no_penalty_fill0","填补 MAE")]
table(["模型","独立假设 t","HAC(5) t","HAC(20) t"],[
 [label,f"{series[n].mean()/(series[n].std()/np.sqrt(len(series[n]))):.2f}",f"{stats[n]['hac5']['t']:.2f}",f"{stats[n]['hac20']['t']:.2f}"] for n,label in stat_names
], [175,108,108,WIDTH-391])
callout("填补 OLS 的 HAC(20) 近似 95% 区间为 [-0.00481, 0.04158]，包含 0；HAC(5) 下也包含 0。当前结果提供弱正向迹象，但不足以确认稳定的样本外预测能力。")
p("方法：Newey-West / Bartlett 权重，以 5 与 20 个交易日滞后作敏感性检查，使用大样本正态近似、未加小样本修正。这是探索性诊断，仍依赖时间序列条件，未校正多模型比较与反复查看测试集。统计不显著不等于证明信号为零。", "small")
p("std_IC 描述逐日相关的离散程度，不是组合风险；IC 也不能直接换算收益。本文未检验分组收益单调性、换手、交易成本及风险暴露，因此不提出可交易性结论。", "small")

title(8,"阶段结论、未解决问题与复现说明","结论围绕已完成的对照，不将预处理改善包装成确定的模型突破。")
h("1. 可以保留的结论")
p("第一，三特征的最初提升不能归因于删除两个特征：同训练样本对照不支持该解释。第二，保留波动率缺失行改善了当前共同评价池上的结果，且影响遍及整个训练期。第三，扩充后的五特征 OLS 在已完成实验中取得最高测试平均 IC；MAE 与填补的收益没有简单叠加。")
h("2. 不能越过的边界")
p("题目明确说明缺失合理，但没有规定处理方式。本报告将删行与填补视为两种需披露的基线口径。若出题者默认填补 OLS，则本阶段尚未实现明确的超基线改进。填补涉及输入处理，是否属于题目限定的改进范围也需澄清；在最终比较中更适合把合理预处理统一固定。")
p("三特征仅为诊断工具；其结果不作为五输入限制下的最终提交模型。2019-2020 年被反复用于观察和形成新假设，已不再是完全未触碰的最终测试期。数值提升、相对基准改善、可靠的绝对预测能力和交易价值应分别评价。")
h("3. 后续工作的优先级")
p("时间允许时，应先确认预处理与评价股票池，固定填补 OLS 为严格比较起点；再按题目提示检验训练标签 rank 变换与低波动样本筛选，并使用预先固定的滚动时间验证。对缺失机制和新增样本贡献的分析，优先于继续堆叠未经解释的损失和惩罚组合。上述工作尚未完成，不计入本报告成果。")
h("4. 代码、数据与结果如何对应")
p("从项目根目录运行 python -m analysis.analysis；逐实验声明位于 analysis/config.py，填写说明见 analysis/manual.md。结果以 result.json 记录配置、系数、指标和逐日序列，summary.csv 汇总，daily_ic.png 展示时间路径。输入填补功能的回归验证记录见 output/feature_fill_regression.json。", "small")
p("本报告生成脚本为 output/pdf/build_exploration_report.py；相邻 exploration_report_assets 保存汇总快照、匹配的结果 JSON、引用图、补充统计及 source_manifest.json。清单记录原始结果相对路径与 SHA-256，供核查，报告生成不修改或重训原实验。", "small")
h("5. 资料与辅助工具使用")
p('[1] JHSS_QUANT远程课题1X.docx：题目、约束与正式 Y 定义。<br/>[2] analysis/results/summary.csv 及对应 result.json、daily_ic.png：本文主表与图 1-3 的来源。<br/>[3] analysis/data.py、training_data.py、models.py、config.py 及测试：处理顺序与实现口径。<br/>[4] Statsmodels，cov_hac 文档：Newey-West / Bartlett 协方差方法说明；本报告使用独立公式计算均值标准误。<br/><link href="https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html" color="#087f7b">www.statsmodels.org / cov_hac 方法文档</link>。', "small")
p("LLM 辅助披露：使用 Codex 辅助代码重构、配置接口设计、测试、结果整理、统计诊断和报告排版。研究者提出并修正实验假设；数值以保存结果和可复算脚本为依据。LLM 的机制解释均作为待验证假说，不替代实证检验。", "small")

PDF=OUT/"五因子线性模型_实验探索与初步结论.pdf"
def footer(c,doc):
    c.setStrokeColor(RULE); c.line(48,39,W-48,39)
    c.setFont("CN",8); c.setFillColor(MUTED)
    c.drawString(48,26,"JHSS_QUANT  |  实验探索与初步结论  |  2026.11.01")
    c.drawRightString(W-48,26,f"{doc.page:02d}")
    c.setFillColor(TEAL); c.rect(0,H-8,W,8,stroke=0,fill=1)
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=38,bottomMargin=50,
                      title="五因子线性模型：实验探索与初步结论",author="项目研究报告",pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/"实验探索报告_内容稿.txt").write_text("\n\n".join(editable),encoding="utf-8")
r=PdfReader(PDF)
print(json.dumps({"pdf":str(PDF),"pages":len(r.pages),"bytes":PDF.stat().st_size,"experiments":len(df),"chars_by_page":[len(p.extract_text()) for p in r.pages]},ensure_ascii=False))
