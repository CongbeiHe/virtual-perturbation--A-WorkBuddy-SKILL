"""VNN1+ vs VNN1- 真实差异表达（肿瘤组织恶性细胞，GSE189175）。

目的：为 Geneformer 虚拟扰动的"方向"提供真实数据锚。

设计（五条并行证据链）：
  A. 主分析   : Tumor & malignant, VNN1 UMI >=1 (pos) vs ==0 (neg)，Wilcoxon
  B. 深度匹配 : 阳性细胞 1:1 最近邻匹配同深度阴性细胞（消除文库大小混杂）
  C. 分样本   : S1_hcc / S2_hcc 各自独立 DE，检验方向可重复性
  D. 连续关联 : VNN1 UMI 数与各基因表达的 Spearman rho + 控制 log(n_counts) 的偏相关
  E. 阈值敏感性: VNN1 >=2 vs ==0（仅 47 细胞，只看方向是否一致）

输出目录 outputs/nashhcc_vnn1_real_de/
"""
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

H5 = r"task/GSE189175/GSE189175_NASHHCC.h5ad"
OUT = Path(r"outputs/nashhcc_vnn1_real_de")
OUT.mkdir(parents=True, exist_ok=True)
MIN_CELLS = 10          # 基因过滤：至少 10 个细胞检出
SEED = 0
rng = np.random.default_rng(SEED)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 读数据
log("读取 h5ad ...")
a = ad.read_h5ad(H5)
log(f"  全部 {a.shape}")

# counts 层为原始整数计数；确认后转 float32 供 scanpy 使用
counts = a.layers["counts"]
log(f"  counts 层类型 {type(counts).__name__}, dtype {counts.dtype}")
if not sp.issparse(counts):
    counts = sp.csr_matrix(counts)
counts = counts.tocsr().astype(np.float32)

mask = ((a.obs["Tissue"].astype(str) == "Tumor") & (a.obs["malignant"].astype(bool))).to_numpy()
log(f"  肿瘤恶性细胞 {mask.sum()}")

sub = ad.AnnData(
    X=counts[mask].copy(),
    obs=a.obs.loc[mask].copy(),
    var=a.var.copy(),
)
sub.var_names_make_unique()
del a, counts

vnn1_idx = int(np.where(sub.var_names == "VNN1")[0][0])
vnn1_raw = np.asarray(sub.X[:, vnn1_idx].todense()).ravel().astype(int)
sub.obs["vnn1_umi"] = vnn1_raw
sub.obs["log_ncounts"] = np.log10(sub.obs["n_counts"].astype(float) + 1)
sub.obs["SampleID"] = sub.obs["SampleID"].astype(str)
log(f"  VNN1+ {int((vnn1_raw>0).sum())} / {len(vnn1_raw)}")

# ---------------------------------------------------------------- 标准化
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)
sub.layers["lognorm"] = sub.X.copy()

# 基因过滤（保留 VNN1 自身以便 sanity check）
detected = np.asarray((sub.X > 0).sum(axis=0)).ravel()
keep = detected >= MIN_CELLS
keep[vnn1_idx] = True
log(f"  检出>={MIN_CELLS} 的基因 {keep.sum()} / {sub.n_vars}")
adt = sub[:, keep].copy()
log(f"  分析矩阵 {adt.shape}")


# ---------------------------------------------------------------- 工具
def run_de(adata, pos_mask, neg_mask, tag, min_cells=MIN_CELLS):
    """在给定 pos/neg 细胞子集上做 Wilcoxon DE，返回全表。"""
    m = pos_mask | neg_mask
    x = adata[m].copy()
    x.obs["grp"] = np.where(pos_mask[m], "pos", "neg")
    n_pos, n_neg = int(pos_mask[m].sum()), int(neg_mask[m].sum())
    if n_pos < 5 or n_neg < 5:
        log(f"  [{tag}] 样本不足 pos={n_pos} neg={n_neg}，跳过")
        return None
    det = np.asarray((x.X > 0).sum(axis=0)).ravel()
    x = x[:, (det >= min_cells) | (x.var_names == "VNN1")].copy()
    sc.tl.rank_genes_groups(x, "grp", groups=["pos"], reference="neg",
                            method="wilcoxon", pts=True, use_raw=False)
    r = sc.get.rank_genes_groups_df(x, group="pos")
    ren = {"names": "gene", "logfoldchanges": "logFC", "pvals": "pval", "pvals_adj": "padj"}
    r = r.rename(columns={k: v for k, v in ren.items() if k in r.columns})
    r = r.drop(columns=[c for c in ["pct_pos", "pct_neg", "logFC_mean", "pct_nz_group", "pct_nz_reference", "pct_nz_ref"] if c in r.columns])
    if "gene" not in r.columns:
        r = r.reset_index().rename(columns={"names": "gene"})
    # 检出比例与均值 log2FC：按基因名合并（rank_genes_groups_df 已按 score 重排，不能按位置赋值）
    Xp = x[x.obs["grp"] == "pos"].X
    Xn = x[x.obs["grp"] == "neg"].X
    pct = pd.DataFrame({
        "gene": np.asarray(x.var_names),
        "pct_pos": np.asarray((Xp > 0).sum(axis=0)).ravel() / max(n_pos, 1) * 100,
        "pct_neg": np.asarray((Xn > 0).sum(axis=0)).ravel() / max(n_neg, 1) * 100,
        "logFC_mean": np.log2(
            (np.asarray(Xp.expm1().mean(axis=0)).ravel() + 1e-9)
            / (np.asarray(Xn.expm1().mean(axis=0)).ravel() + 1e-9)),
    })
    pct["pct_pos"] = pct["pct_pos"].round(2)
    pct["pct_neg"] = pct["pct_neg"].round(2)
    r = r.merge(pct, on="gene", how="left")
    r["n_pos"], r["n_neg"] = n_pos, n_neg
    r["method"] = tag
    r = r[["gene", "logFC", "logFC_mean", "pval", "padj", "pct_pos", "pct_neg", "n_pos", "n_neg", "method"]]
    log(f"  [{tag}] pos={n_pos} neg={n_neg} 测试基因 {len(r)}  "
        f"padj<0.05 {int((r.padj<0.05).sum())}  padj<0.01 {int((r.padj<0.01).sum())}")
    return r


# ---------------------------------------------------------------- A 主分析
log("\n[A] 主分析：VNN1>=1 vs 0")
pos = vnn1_raw >= 1
neg = vnn1_raw == 0
de_main = run_de(adt, pos, neg, "main")

# ---------------------------------------------------------------- C 分样本
log("\n[C] 分样本 DE")
de_by = {}
for s in ["S2_hcc", "S1_hcc"]:
    sm = (adt.obs["SampleID"] == s).to_numpy()
    r = run_de(adt, pos & sm, neg & sm, f"sample_{s}")
    if r is not None:
        de_by[s] = r

# ---------------------------------------------------------------- B 深度匹配
log("\n[B] 深度匹配（1:1 最近邻 n_counts）")
pos_idx = np.where(pos)[0]
neg_idx = np.where(neg)[0]
depth = adt.obs["log_ncounts"].to_numpy().reshape(-1, 1)
nn = NearestNeighbors(n_neighbors=min(len(neg_idx), 200)).fit(depth[neg_idx])
nn_idx = nn.kneighbors(depth[pos_idx], return_distance=False)
used, matched = set(), []
for row in nn_idx:
    for j in row:
        if j not in used:
            used.add(j)
            matched.append(neg_idx[j])
            break
matched = np.array(matched)
pos_m = np.zeros(adt.n_obs, bool)
neg_m = np.zeros(adt.n_obs, bool)
pos_m[pos_idx] = True
neg_m[matched] = True
log(f"  匹配成功 {len(matched)} 对；深度中位 阳性 {np.median(adt.obs['n_counts'].to_numpy()[pos_idx]):.0f}"
    f" vs 匹配阴性 {np.median(adt.obs['n_counts'].to_numpy()[matched]):.0f}")
de_match = run_de(adt, pos_m, neg_m, "depth_matched")

# ---------------------------------------------------------------- E 严格阈值
log("\n[E] 阈值敏感性：VNN1>=2 vs 0")
de_strict = run_de(adt, vnn1_raw >= 2, vnn1_raw == 0, "strict_ge2")

# ---------------------------------------------------------------- D 连续关联
log("\n[D] 连续关联（Spearman rho + 偏相关，控制 log n_counts）")
X = adt.X.toarray().astype(np.float32)
gnames = adt.var_names.to_numpy()
ncounts = adt.obs["log_ncounts"].to_numpy().astype(np.float64)

# 对 X 与 VNN1 共同残差化，再算 Pearson = 偏相关
def residualize(mat, cov):
    cov = np.column_stack([np.ones(len(cov)), cov])
    beta, *_ = np.linalg.lstsq(cov, mat, rcond=None)
    return mat - cov @ beta

v = vnn1_raw.astype(np.float64)
Xr = residualize(X, ncounts)
vr = residualize(v.reshape(-1, 1), ncounts).ravel()

rx = stats.rankdata(X, axis=0)
rv = stats.rankdata(v)
rx = (rx - rx.mean(0)) / (rx.std(0) + 1e-12)
rv = (rv - rv.mean()) / (rv.std() + 1e-12)
rho = (rv[:, None] * rx).mean(0)

def corr_p(r, n):
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), n - 2)

p_rho = corr_p(rho, len(v))

Xc = Xr - Xr.mean(0)
vc = vr - vr.mean()
denom = np.sqrt((Xc ** 2).sum(0) * (vc ** 2).sum()) + 1e-12
pr = (vc @ Xc) / denom
p_pr = corr_p(pr, len(v) - 3)

corr = pd.DataFrame({
    "gene": gnames, "spearman_rho": rho, "pval_rho": p_rho,
    "partial_rho": pr, "pval_partial": p_pr,
    "n_cells": len(v), "n_vnn1_pos": int((v > 0).sum()),
})
# 用 BH 校正
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)

corr["padj_rho"] = bh(corr["pval_rho"])
corr["padj_partial"] = bh(corr["pval_partial"])
corr = corr.sort_values("partial_rho", ascending=False)
log(f"  |partial_rho| 最大 {np.abs(corr.partial_rho).max():.3f}；"
    f"padj_partial<0.05 {int((corr.padj_partial<0.05).sum())} 个")

# ---------------------------------------------------------------- 汇总
log("\n汇总一致性表 ...")
merged = de_main[["gene", "logFC", "logFC_mean", "padj", "pct_pos", "pct_neg"]].rename(
    columns={"logFC": "logFC_main", "logFC_mean": "logFC_main_mean", "padj": "padj_main"})
if de_match is not None:
    merged = merged.merge(
        de_match[["gene", "logFC", "logFC_mean", "padj"]].rename(
            columns={"logFC": "logFC_matched", "logFC_mean": "logFC_matched_mean", "padj": "padj_matched"}),
        on="gene", how="left")
if de_strict is not None:
    merged = merged.merge(
        de_strict[["gene", "logFC", "padj"]].rename(columns={"logFC": "logFC_strict", "padj": "padj_strict"}),
        on="gene", how="left")
merged = merged.merge(corr[["gene", "spearman_rho", "partial_rho", "padj_partial"]], on="gene", how="left")

for s, r in de_by.items():
    merged = merged.merge(
        r[["gene", "logFC"]].rename(columns={"logFC": f"logFC_{s}"}), on="gene", how="left")

# 方向一致性打分（各证据链符号是否一致）
def sign(x):
    return np.sign(np.nan_to_num(np.asarray(x, float)))

col_sets = [["logFC_main", "logFC_matched", "partial_rho"]]
if "logFC_sample_S2_hcc" in merged:
    col_sets[0].append("logFC_sample_S2_hcc")
    if "logFC_sample_S1_hcc" in merged:
        col_sets[0].append("logFC_sample_S1_hcc")
cols = [c for c in col_sets[0] if c in merged]
S = np.column_stack([sign(merged[c]) for c in cols])
merged["n_agree"] = (S == 1).sum(1) + (S == -1).sum(1)
merged["all_up"] = (S == 1).all(1)
merged["all_down"] = (S == -1).all(1)
merged["direction"] = np.where(merged.all_up, "UP", np.where(merged.all_down, "DOWN", "MIXED"))

# 主排序：主分析显著 + 一致
merged["neglog10_padj"] = -np.log10(merged["padj_main"].clip(lower=1e-300))
merged = merged.sort_values(["padj_main", "logFC_main"], ascending=[True, False])

de_main.to_csv(OUT / "de_main_vnn1pos_vs_neg.csv", index=False)
if de_match is not None:
    de_match.to_csv(OUT / "de_depth_matched.csv", index=False)
if de_strict is not None:
    de_strict.to_csv(OUT / "de_strict_ge2.csv", index=False)
for s, r in de_by.items():
    r.to_csv(OUT / f"de_{s}.csv", index=False)
corr.to_csv(OUT / "continuous_correlation.csv", index=False)
merged.to_csv(OUT / "de_consensus.csv", index=False)
log(f"  已写出 6 张表到 {OUT}")

# ---------------------------------------------------------------- 摘要
log("\n=== 主分析 top20（按 padj）===")
show = ["gene", "logFC", "logFC_mean", "padj", "pct_pos", "pct_neg"]
log(de_main.head(20)[show].to_string(index=False))

log("\n=== 一致性 UP/DOWN 计数（padj_main<0.05）===")
sig = merged[merged.padj_main < 0.05]
log(sig.direction.value_counts().to_string())

log("\n=== 一致 UP 且主分析显著 top15 ===")
log(sig[sig.direction == "UP"].head(15)[
    ["gene", "logFC_main", "padj_main", "logFC_matched", "partial_rho", "pct_pos", "pct_neg"]
].to_string(index=False))

log("\n=== 一致 DOWN 且主分析显著 top15 ===")
log(sig[sig.direction == "DOWN"].head(15)[
    ["gene", "logFC_main", "padj_main", "logFC_matched", "partial_rho", "pct_pos", "pct_neg"]
].to_string(index=False))

log("\n=== VNN1 自身（sanity check，分组依据，循环论证）===")
log(de_main[de_main.gene == "VNN1"].to_string(index=False))

log("\n完成。")
