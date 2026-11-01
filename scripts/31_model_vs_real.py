"""用真实 DE 给 Geneformer 虚拟扰动的输出打分，并产出高置信基因表。

输入:
  outputs/nashhcc_vnn1_real_de/de_consensus.csv      真实 DE（主/匹配/严格/分样本/连续）
  outputs/nashhcc_tumor_isp_stats/vnn1_oe_gene_shifts.csv   Geneformer 余弦幅度
  outputs/nashhcc_tumor_vnn1_analysis/vnn1_oe_direction.csv Geneformer 方向（秩校准轴）

分析:
  1. 幅度校验：真实 DE 显著基因的模型余弦位移是否更大（AUC）
  2. 方向校验：模型 dir_z 符号 vs 真实 DE logFC 符号（一致率、AUC、Spearman）
  3. 分层：按真实 DE 效应量与检出率分箱，看模型方向准确率如何变化
  4. 三重证据基因表：真实 DE 显著 + 深度匹配同向 + 连续偏相关同向 + 模型同向

输出: outputs/nashhcc_vnn1_real_de/model_vs_real/
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = Path(r"outputs/nashhcc_vnn1_real_de/model_vs_real")
OUT.mkdir(parents=True, exist_ok=True)

de = pd.read_csv(r"outputs/nashhcc_vnn1_real_de/de_consensus.csv")
gs = pd.read_csv(r"outputs/nashhcc_tumor_isp_stats/vnn1_oe_gene_shifts.csv")
dr = pd.read_csv(r"outputs/nashhcc_tumor_vnn1_analysis/vnn1_oe_direction.csv")

print(f"真实 DE {len(de)} 基因；模型幅度 {len(gs)}；模型方向 {len(dr)}")

# 模型幅度表：Cosine_sim_mean 越小 = 位移越大
gs = gs.rename(columns={"Affected_gene_name": "gene"})
gs["shift"] = 1.0 - gs["Cosine_sim_mean"]

# ---------------------------------------------------------------- 1 幅度校验
print("\n=== 1. 模型余弦位移 能否区分「真实 DE 显著」 ===")
m = de.merge(gs[["gene", "shift", "N_Detections"]], on="gene", how="inner")
m = m[m["N_Detections"] >= 20]
sig = m["padj_main"] < 0.05
print(f"交集 {len(m)} 基因，其中真实 DE 显著 {int(sig.sum())} 个")
if sig.any() and (~sig).any():
    auc = roc_auc_score(sig, m["shift"])
    print(f"  AUC = {auc:.3f}  (>0.5 表示显著基因的模型位移确实更大)")
    print(f"  显著基因 shift 中位 {m.loc[sig,'shift'].median():.5f} vs 不显著 {m.loc[~sig,'shift'].median():.5f}")
    u = stats.mannwhitneyu(m.loc[sig, "shift"], m.loc[~sig, "shift"], alternative="greater")
    print(f"  Mann-Whitney p = {u.pvalue:.2e}")

# ---------------------------------------------------------------- 2 方向校验
print("\n=== 2. 模型方向 vs 真实 DE 方向 ===")
d = dr.merge(de[["gene", "logFC_main", "padj_main", "logFC_matched", "partial_rho", "pct_pos", "pct_neg"]],
             on="gene", how="inner")
d = d.dropna(subset=["logFC_main", "dir_z"])
print(f"交集 {len(d)} 基因")

# 只用真实 DE 显著的基因做方向一致性检验（否则"真实方向"本身是噪声）
d_sig = d[(d.padj_main < 0.05) & (d.logFC_main.abs() > 0.25)].copy()
print(f"真实 DE 显著且 |logFC|>0.25 的 {len(d_sig)} 个")

if len(d_sig) >= 20:
    agree = (np.sign(d_sig.dir_z) == np.sign(d_sig.logFC_main))
    rate = agree.mean()
    bt = stats.binomtest(int(agree.sum()), len(agree), 0.5, alternative="greater")
    print(f"  方向一致率 {rate:.1%} ({int(agree.sum())}/{len(agree)})，二项检验 p = {bt.pvalue:.2e}")
    auc_d = roc_auc_score((d_sig.logFC_main > 0).astype(int), d_sig.dir_z)
    print(f"  方向 AUC = {auc_d:.3f}  (0.5 = 随机)")
    rho = stats.spearmanr(d_sig.dir_z, d_sig.logFC_main)
    print(f"  Spearman(dir_z, logFC) = {rho.statistic:.3f}, p = {rho.pvalue:.2e}")

    # 分层：模型 |dir_z| 越大，一致率是否越高
    print("\n  按 |dir_z| 分位分层：")
    d_sig["abs_z_bin"] = pd.qcut(d_sig.dir_z.abs(), min(4, len(d_sig) // 10), labels=False, duplicates="drop")
    for b, g in d_sig.groupby("abs_z_bin"):
        a = (np.sign(g.dir_z) == np.sign(g.logFC_main)).mean()
        print(f"    第{int(b)+1}层 |dir_z| 中位 {g.dir_z.abs().median():6.2f}  n={len(g):3d}  一致率 {a:.1%}")

# 全部基因（不筛显著）的相关
if len(d) >= 20:
    rho_all = stats.spearmanr(d.dir_z, d.logFC_main)
    print(f"\n  全部交集基因 Spearman = {rho_all.statistic:.3f} (p={rho_all.pvalue:.2e})")

# ---------------------------------------------------------------- 3 三重证据
print("\n=== 3. 三重证据基因 ===")
# 真实 DE 显著 + 深度匹配同向 + 偏相关同向 + 模型同向
c = de.copy()
c = c.merge(dr[["gene", "dir_z", "direction"]].rename(columns={"direction": "model_dir"}), on="gene", how="left")
c = c.merge(gs[["gene", "shift"]], on="gene", how="left")

cond_real = (c.padj_main < 0.05)
cond_match = np.sign(c.logFC_matched) == np.sign(c.logFC_main)
cond_partial = np.sign(c.partial_rho) == np.sign(c.logFC_main)
cond_model = np.sign(c.dir_z) == np.sign(c.logFC_main)

c["real_sig"] = cond_real
c["matched_agree"] = cond_match
c["partial_agree"] = cond_partial
c["model_agree"] = cond_model
c["evidence_score"] = cond_real.astype(int) + cond_match.astype(int) + cond_partial.astype(int) + cond_model.astype(int)
c["confidence"] = np.select(
    [c.evidence_score >= 4, c.evidence_score == 3, c.evidence_score == 2],
    ["A_high", "B_medium", "C_low"], default="D_reject")

c = c.sort_values(["evidence_score", "padj_main"], ascending=[False, True])
c.to_csv(OUT / "consensus_gene_table.csv", index=False)

print(c.confidence.value_counts().sort_index().to_string())
print(f"\nA 级（四链全一致）{int((c.confidence=='A_high').sum())} 个，前 20：")
show = ["gene", "logFC_main", "padj_main", "logFC_matched", "partial_rho", "dir_z", "pct_pos", "pct_neg"]
print(c[c.confidence == "A_high"].head(20)[show].round(3).to_string(index=False))

print(f"\nB 级 {int((c.confidence=='B_medium').sum())} 个，前 10：")
print(c[c.confidence == "B_medium"].head(10)[show].round(3).to_string(index=False))

# 与既往 Geneformer 稳健基因表交集，检查已知轴
known = ["CYP2E1", "CYP3A5", "RBP4", "SLC30A10", "KEAP1", "HDAC3", "RELA", "INSIG2", "TXNRD1",
         "AKR1C3", "CPVL", "ALB", "FOXA1", "CYP3A4", "ADH1B", "CPS1", "NNMT", "PID1"]
print("\n=== 已知/关注基因在真实 DE 中的表现 ===")
k = c[c.gene.isin(known)][["gene", "logFC_main", "padj_main", "pct_pos", "pct_neg",
                           "logFC_matched", "partial_rho", "dir_z", "confidence"]]
print(k.round(3).to_string(index=False))

# 保存给后续绘图
summary = {
    "n_overlap_direction": len(d),
    "n_sig_direction": len(d_sig) if len(d_sig) else 0,
    "direction_agree_rate": float(rate) if len(d_sig) >= 20 else np.nan,
    "direction_auc": float(auc_d) if len(d_sig) >= 20 else np.nan,
    "amplitude_auc": float(auc) if sig.any() and (~sig).any() else np.nan,
}
pd.Series(summary).to_csv(OUT / "validation_summary.csv")
print(f"\n-> {OUT}")
