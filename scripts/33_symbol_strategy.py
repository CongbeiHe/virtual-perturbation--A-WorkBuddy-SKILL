"""测试"只加符号、不看 FC"的混合方案：不同加符号策略的准确率。

策略:
  S1 sign(dir_z)                     基线
  S2 sign(dir_score - null_mean)     净位移（扣除全局公共分量）
  S3 S1 但仅 |dir_z| 分位 top 层      自信子集
  S4 S2 且 dir_z 与 null_z 反号       净信号明确子集

标签: 真实 DE logFC_main 符号（仅在可信子集上评估：两组 pct>=2%、padj<0.05）
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(r"outputs/nashhcc_vnn1_real_de/model_vs_real")
OUT.mkdir(parents=True, exist_ok=True)

de = pd.read_csv(r"outputs/nashhcc_vnn1_real_de/de_consensus.csv")
dr = pd.read_csv(r"outputs/nashhcc_tumor_vnn1_analysis/vnn1_oe_direction.csv")

d = dr.merge(de[["gene", "logFC_main", "padj_main", "pct_pos", "pct_neg"]], on="gene", how="inner")
d = d[(d.pct_pos >= 2) & (d.pct_neg >= 2)]
print(f"交集 {len(d)} 基因")

# 标签可信子集：真实 DE 显著（方向本身有意义）
d["sig"] = d.padj_main < 0.05
lab = d[d.sig].copy()
print(f"标签可信子集（真实显著）{len(lab)}: {lab.gene.tolist()}")

y = np.sign(d.logFC_main)


def acc(mask, s):
    if mask.sum() == 0:
        return np.nan, 0
    return float((np.sign(s[mask]) == y[mask]).mean()), int(mask.sum())


d["s1"] = np.sign(d.dir_z)
d["s2"] = np.sign(d.dir_score - d.null_mean)

print("\n--- 全部 237 基因（未筛标签显著性，参考） ---")
for name, s in [("S1 sign(dir_z)", d.s1), ("S2 净位移", d.s2)]:
    a, n = acc(np.ones(len(d), bool), s)
    p = stats.binomtest(int((np.sign(s) == y).sum()), len(d), 0.5).pvalue
    print(f"  {name:18s} 一致率 {a:.1%}  n={n}  p={p:.3f}")

print("\n--- 仅真实显著子集（label 可信）---")
if len(lab) > 0:
    lab = lab.assign(s1=np.sign(lab.dir_z), s2=np.sign(lab.dir_score - lab.null_mean))
    for name, col in [("S1 sign(dir_z)", "s1"), ("S2 净位移", "s2")]:
        a = (np.sign(lab[col]) == np.sign(lab["logFC_main"])).mean()
        print(f"  {name:18s} 一致率 {a:.1%}  n={len(lab)}")

print("\n--- S3：仅 |dir_z| 分位 top 层（自信子集，全部基因） ---")
qbin = pd.qcut(d.dir_z.abs(), 4, labels=False, duplicates="drop")
for b, g in d.groupby(qbin):
    a = (g.s1 == np.sign(g.logFC_main)).mean()
    print(f"  层{int(b)+1} |dir_z| 中位 {g.dir_z.abs().median():7.2f} n={len(g):3d} 一致率 {a:.1%}")

print("\n--- S4：净信号明确（dir_z 与 null_z 反号） ---")
m4 = np.sign(d.dir_z) != np.sign(d.null_z)
a4 = (d.s2[m4] == y[m4]).mean()
print(f"  n={int(m4.sum())}  一致率 {a4:.1%}")
m4sig = m4 & d.sig
if m4sig.sum():
    a4s = (d.s2[m4sig] == y[m4sig]).mean()
    print(f"  其中真实显著 {int(m4sig.sum())} 个，一致率 {a4s:.1%}")

# 组合策略：S2 且 |dir_z| top 25%
print("\n--- 组合：S2 且 |dir_z| top25% ---")
m5 = (d.dir_z.abs() >= d.dir_z.abs().quantile(0.75))
a5 = (d.s2[m5] == y[m5]).mean()
print(f"  n={int(m5.sum())}  一致率 {a5:.1%}")
m5sig = m5 & d.sig
if m5sig.sum():
    print(f"  其中真实显著 {int(m5sig.sum())} 个：{d.loc[m5sig,'gene'].tolist()}")

# 真实显著的 3 个基因在各策略下的表现
print("\n--- 3 个真实显著基因（CYP2E1/RBP4/CYP3A5 类）逐个 ---")
cols = ["gene", "logFC_main", "dir_z", "null_z", "s1", "s2", "sig"]
print(d[d.sig][cols].round(3).to_string(index=False))
