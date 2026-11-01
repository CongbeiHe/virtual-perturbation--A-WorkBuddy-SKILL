"""
步骤 5：分析虚拟扰动结果

输入是 04 脚本产出的统计 CSV，核心指标有四个：

| 字段 | 含义 | 怎么读 |
|------|------|--------|
| `Test_avg_shift` | 扰动后的平均余弦相似度 | **越小 = 影响越大**（1.0 表示毫无变化） |
| `Impact_component` | 混合模型中「有影响」成分的后验权重 | 越接近 1 越可能真有效应 |
| `N_Detections` | 该基因在多少个细胞中被检测到 | 越大说明结论越稳健 |
| `Impact_component_percent` | 上述权重的百分位 | 用于快速排序 |

本脚本输出：
1. 影响最大的 Top-N 基因表格（CSV）
2. 影响分布直方图
3. Top-N 条形图
4. 按基因家族（核糖体 / 线粒体 / MHC 等）的功能归类统计

用法：
    python scripts/05_analyze_perturbation.py
    python scripts/05_analyze_perturbation.py --input outputs/isp_stats/pbmc3k_n20.csv --top 40
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for _f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 基因家族归类
# 无需外部数据库的轻量分类：按基因名规则归类，用于快速看出功能倾向
FAMILY_RULES = [
    ("核糖体蛋白 (RPL/RPS)",   r"^RP[LS]\d"),
    ("线粒体 (MT-)",           r"^MT-"),
    ("MHC/HLA 抗原呈递",       r"^HLA-|^B2M$"),
    ("翻译延伸因子",           r"^EEF\d|^EIF\d"),
    ("核糖体生物发生/核仁",     r"^NPM|^NCL|^NOL|^GAR1|^DKC1|^FBL"),
    ("热休克蛋白 (HSP)",       r"^HSP|^DNAJ"),
    ("组蛋白 (HIST)",          r"^HIST|^H2A|^H2B|^H3-|^H4C"),
    ("细胞骨架/肌动蛋白",       r"^ACT|^TMSB|^TUB|^MYL|^PFN|^CFL|^ARPC|^COTL"),
    ("T 细胞相关",             r"^CD3|^LCK$|^ZAP70$|^IL7R$|^CD2$|^CD28$|^TRBC|^CD8|^CD4$"),
    ("B 细胞相关",             r"^CD79|^MS4A1$|^PAX5$|^CD19$|^TCL1A$|^IGHM|^CD74$"),
    ("髓系/单核相关",           r"^LYZ$|^S100A|^FCGR|^CD14$|^SPI1$|^CEBP|^FCN1$|^CST3$"),
    ("NK/细胞毒",              r"^NKG7$|^GNLY$|^KLRD1$|^PRF1$|^GZMB|^GZMA|^KLR"),
    ("转录因子 (常见)",         r"^FOS|^JUN|^MYC$|^NFKB|^STAT|^IRF|^EBF1$|^TCF"),
    ("趋化因子/细胞因子",       r"^CCL|^CXCL|^IL\d|^TNF$|^IFN"),
]


def classify(name: str) -> str:
    if not isinstance(name, str):
        return "其他"
    for fam, pat in FAMILY_RULES:
        if re.search(pat, name, re.IGNORECASE):
            return fam
    return "其他"


def main():
    ap = argparse.ArgumentParser(description="虚拟扰动结果分析")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "outputs" / "isp_stats" / "pbmc3k_n20.csv"))
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "outputs" / "isp_analysis"))
    ap.add_argument("--prefix", default=None, help="输出前缀，默认取输入文件名")
    ap.add_argument("--top", type=int, default=30, help="条形图展示的基因数")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        # 回退到任意可用的 stats 文件
        cands = sorted((PROJECT_ROOT / "outputs" / "isp_stats").glob("*.csv"))
        if not cands:
            raise SystemExit(
                f"找不到结果文件 {src}\n请先运行: python scripts/04_in_silico_perturbation.py"
            )
        src = cands[-1]
        print(f"[info] 使用找到的结果文件: {src.name}")

    prefix = args.prefix or src.stem
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    print(f"读入: {src}  形状 {df.shape}")
    print(f"列: {list(df.columns)}\n")

    # 兼容不同模式的列名
    shift_col = "Test_avg_shift"
    if shift_col not in df.columns:
        raise SystemExit(f"结果里没有 {shift_col} 列，可能统计模式不支持本分析")

    df = df.dropna(subset=[shift_col]).copy()
    df["Impact"] = 1.0 - df[shift_col]          # 影响幅度（越大越关键）
    df["Family"] = df["Gene_name"].apply(classify)
    df = df.sort_values(shift_col)

    print(f"参与统计的基因: {len(df)}")
    print(f"影响幅度 (1 - cos_sim):  均值 {df['Impact'].mean():.5f}   "
          f"最大 {df['Impact'].max():.5f}   最小 {df['Impact'].min():.5f}\n")

    # ---------------- 1) Top-N 表格 ----------------
    cols = [c for c in ["Gene_name", "Ensembl_ID", "Gene", shift_col,
                        "Impact", "Impact_component", "N_Detections", "Family"]
            if c in df.columns]
    top_df = df.head(args.top)[cols]
    out_csv = out_dir / f"{prefix}_top{args.top}_genes.csv"
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Top-{args.top} 基因表已保存: {out_csv}")
    print(f"\n影响最大的 15 个基因:")
    print(top_df.head(15).to_string(index=False))

    # ---------------- 2) 影响分布直方图 ----------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.hist(df["Impact"], bins=40, color="#3B6FB6", alpha=0.85, edgecolor="white", linewidth=0.4)
    med = df["Impact"].median()
    ax.axvline(med, color="#D9534F", linestyle="--", linewidth=1.4,
               label=f"中位数 {med:.5f}")
    ax.set_xlabel("影响幅度（1 - 余弦相似度）")
    ax.set_ylabel("基因数")
    ax.set_title(f"虚拟扰动影响分布 · {len(df)} 个基因", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p_hist = out_dir / f"{prefix}_impact_distribution.png"
    fig.savefig(p_hist, bbox_inches="tight")
    plt.close(fig)
    print(f"\n影响分布图: {p_hist}")

    # ---------------- 3) Top-N 条形图 ----------------
    topn = df.head(args.top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(5, args.top * 0.24)), dpi=140)
    colors = ["#D9534F" if f == "核糖体蛋白 (RPL/RPS)" else "#3B6FB6" for f in topn["Family"]]
    ax.barh(topn["Gene_name"], topn["Impact"], color=colors, height=0.75)
    ax.set_xlabel("影响幅度（1 - 余弦相似度）")
    ax.set_title(f"扰动影响最大的 {args.top} 个基因\n"
                 f"（红 = 核糖体蛋白；{src.name}）", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    p_bar = out_dir / f"{prefix}_top{args.top}_bar.png"
    fig.savefig(p_bar, bbox_inches="tight")
    plt.close(fig)
    print(f"Top-{args.top} 条形图: {p_bar}")

    # ---------------- 4) 基因家族统计 ----------------
    fam = (df.groupby("Family")
             .agg(基因数=("Impact", "size"),
                  平均影响=("Impact", "mean"),
                  最大影响=("Impact", "max"))
             .sort_values("平均影响", ascending=False))
    fam["占比"] = (fam["基因数"] / fam["基因数"].sum()).map(lambda x: f"{x:.1%}")
    print(f"\n按基因家族归类（按平均影响排序）:")
    print(fam.to_string())
    fam_csv = out_dir / f"{prefix}_family_summary.csv"
    fam.to_csv(fam_csv, encoding="utf-8-sig")
    print(f"\n家族汇总: {fam_csv}")

    # 家族平均影响对比图（只画基因数 >= 5 的家族，避免小样本噪声）
    fam_plot = fam[fam["基因数"] >= 5]
    if len(fam_plot) >= 2:
        fig, ax = plt.subplots(figsize=(8.5, max(4, len(fam_plot) * 0.42)), dpi=140)
        fp = fam_plot.iloc[::-1]
        ax.barh(fp.index, fp["平均影响"], color="#5B8FF9", height=0.7)
        ax.set_xlabel("平均影响幅度")
        ax.set_title(f"各基因家族的平均扰动影响（基因数 ≥ 5）", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        p_fam = out_dir / f"{prefix}_family_impact.png"
        fig.savefig(p_fam, bbox_inches="tight")
        plt.close(fig)
        print(f"家族对比图: {p_fam}")

    # ---------------- 5) 稳健性提示 ----------------
    print("\n" + "=" * 62)
    print("解读要点")
    print("=" * 62)
    if "N_Detections" in df.columns:
        n_max = int(df["N_Detections"].max())
        print(f"- N_Detections 最大值 = {n_max}，说明本实验覆盖了 {n_max} 个细胞的检测")
        print(f"  结论稳健性随该值升高；若普遍很低（如 1~2），建议用更多细胞重跑")
    top1 = df.iloc[0]
    print(f"- 影响最大的基因是 {top1['Gene_name']}（{top1['Ensembl_ID']}），"
          f"影响幅度 {top1['Impact']:.5f}")
    print("- 高表达基因（核糖体蛋白、TMSB4X 等）通常排名靠前，这是因为 Geneformer")
    print("  的 rank value encoding 对序列前段的基因更敏感——属于模型机制的合理体现。")
    print("- 想做「目标状态迁移」分析，可改用 --stats-mode goal_state_shift，")
    print("  并先用 EmbExtractor.get_state_embs() 算好各状态的 embedding 位置。")


if __name__ == "__main__":
    main()
