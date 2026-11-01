"""控制检出数后，检验模型余弦位移与真实 DE 的关系。

背景：shift = 1 - cos 与 N_Detections 强负相关（低检出基因 embedding 噪声大），
而真实 DE 显著基因多为中等检出。未分层时 AUC=0.294 可能是这一混杂造成。

做法：
  1. 按 N_Detections 十分位分层，箱内算 AUC，再按样本量加权平均（协变量校正 AUC）
  2. 检出数匹配子集（同一区间内比较）
  3. 在"高检出且 DE 显著"的双重约束下，检验 |logFC| 与 shift 的秩相关
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = Path(r"outputs/nashhcc_vnn1_real_de/model_vs_real")
OUT.mkdir(parents=True, exist_ok=True)

de = pd.read_csv(r"outputs/nashhcc_vnn1_real_de/de_consensus.csv")
gs = pd.read_csv(r"outputs/nashhcc_tumor_isp_stats/vnn1_oe_gene_shifts.csv").rename(
    columns={"Affected_gene_name": "gene"})
gs["shift"] = 1.0 - gs["Cosine_sim_mean"]

m = de.merge(gs[["gene", "shift", "N_Detections"]], on="gene", how="inner")
m["sig"] = m.padj_main < 0.05
print(f"交集 {len(m)}；显著 {int(m.sig.sum())}")

# 稀疏假象过滤：两组检出率都过低（<2%）时 logFC 不可信
q = m[(m.pct_pos >= 2) & (m.pct_neg >= 2)].copy()
print(f"检出率过滤后 {len(q)}（去除 pct<2% 的极端稀疏 {len(m)-len(q)} 个）；显著 {int(q.sig.sum())}")
m = q

print("\n=== 未分层 ===")
print(f"  AUC {roc_auc_score(m.sig, m["shift"]):.3f}")

# ---------------- 分层 AUC
print("\n=== 按 N_Detections 十分位分层 ===")
m["bin"] = pd.qcut(m.N_Detections, 10, labels=False, duplicates="drop")
rows = []
for b, g in m.groupby("bin"):
    if g.sig.nunique() < 2 or len(g) < 30:
        continue
    auc_b = roc_auc_score(g.sig, g["shift"])
    rows.append({"bin": int(b), "n": len(g), "n_sig": int(g.sig.sum()),
                 "det_median": float(g.N_Detections.median()),
                 "shift_median": float(g["shift"].median()), "auc": auc_b})
b_df = pd.DataFrame(rows)
print(b_df.round(3).to_string(index=False))
w = b_df.n / b_df.n.sum()
print(f"\n  加权平均分层 AUC = {(b_df.auc * w).sum():.3f}")
print(f"  各层 AUC 中位 = {b_df.auc.median():.3f}，范围 [{b_df.auc.min():.3f}, {b_df.auc.max():.3f}]")

# ---------------- 检出数匹配子集
print("\n=== 检出数匹配子集（20-200，噪声与信号都较均衡的区间）===")
sub = m[(m.N_Detections >= 20) & (m.N_Detections <= 200)]
if sub.sig.nunique() == 2 and len(sub) > 50:
    print(f"  n={len(sub)}，显著 {int(sub.sig.sum())}，AUC={roc_auc_score(sub.sig, sub['shift']):.3f}")
    print(f"  显著 shift 中位 {sub.loc[sub.sig,'shift'].median():.5f} vs 不显著 {sub.loc[~sub.sig,'shift'].median():.5f}")

print("\n=== 高检出区间（>=500）===")
hi = m[m.N_Detections >= 500]
if hi.sig.nunique() == 2 and len(hi) > 50:
    print(f"  n={len(hi)}，显著 {int(hi.sig.sum())}，AUC={roc_auc_score(hi.sig, hi['shift']):.3f}")

# ---------------- 效应量 vs 位移（仅在显著基因内）
print("\n=== 显著基因内部：|logFC| 与 shift 的关系 ===")
sg = m[m.sig]
if len(sg) > 30:
    r = stats.spearmanr(sg.logFC_main.abs(), sg["shift"])
    print(f"  n={len(sg)}  Spearman={r.statistic:.3f}  p={r.pvalue:.2e}")
    sg2 = sg[sg.N_Detections >= 100]
    if len(sg2) > 20:
        r2 = stats.spearmanr(sg2.logFC_main.abs(), sg2["shift"])
        print(f"  限 N>=100: n={len(sg2)}  Spearman={r2.statistic:.3f}  p={r2.pvalue:.2e}")

# ---------------- 检出的稳健基因（既往报告用 N>=100）
print("\n=== 既往「稳健」筛选（N>=100）在真实 DE 中的命中率 ===")
rb = m[m.N_Detections >= 100]
print(f"  N>=100 的基因 {len(rb)}，其中真实 DE 显著 {int(rb.sig.sum())} ({rb.sig.mean():.1%})")
print(f"  全体显著比例 {m.sig.mean():.1%}  → 富集倍数 {rb.sig.mean()/m.sig.mean():.2f}x")
tab = pd.crosstab(rb.sig, rb.N_Detections >= 500)
if tab.shape == (2, 2):
    print(f"  列联表:\n{tab.to_string()}")

b_df.to_csv(OUT / "amplitude_layered_auc.csv", index=False)
print(f"\n-> {OUT/'amplitude_layered_auc.csv'}")
