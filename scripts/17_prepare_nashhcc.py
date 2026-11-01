"""
GSE189175 NASH-HCC 数据集：抽取 malignant 细胞并转成 Geneformer 可接受格式。

数据现状：
  - X = log1p(CP10K)（对数归一化）
  - layers["counts"] = 原始整数计数（已验证 100% 整数）
  - var 索引是基因符号（symbol），需映射为 Ensembl ID
  - obs 有 cell_type（6 类）、malignant（bool）、SampleID/Tissue/Condition/Patient、n_counts
  - obsm["X_umap"] 保留供可视化参考

处理：
  1. 子集 malignant == True（3844 细胞，全部来自 Hepatocyte）
  2. X 用 layers["counts"]（int32），n_counts 重算
  3. var 索引转 Ensembl ID；无法映射的基因丢弃，symbol 存入 feature_name
  4. obs 元数据全部转字符串
  5. 保存到 data/nashhcc_counts/malignant/GSE189175_malignant.h5ad
"""

import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(r"E:\Geneformer")
SRC = ROOT / "task" / "GSE189175" / "GSE189175_NASHHCC.h5ad"
DST_DIR = ROOT / "data" / "nashhcc_counts" / "malignant"
KEEP_ATTRS = ["cell_type", "malignant", "SampleID", "Patient", "Tissue",
              "Condition", "patient", "tissue", "group"]
VNN1 = "ENSG00000112299"


def log(m=""):
    print(m, flush=True)


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log(f"读取 {SRC}")
    a = ad.read_h5ad(SRC)
    log(f"  原始: {a.shape[0]} 细胞 × {a.shape[1]} 基因")

    # ---------------- 1. 子集 malignant ----------------
    mask = a.obs["malignant"].astype(bool).to_numpy()
    log(f"  malignant 细胞 {mask.sum()} / {len(mask)}")
    b = a[mask].copy()
    log(f"  子集后: {b.shape[0]} 细胞")

    # ---------------- 2. 计数矩阵 ----------------
    counts = b.layers["counts"]
    if sp.issparse(counts):
        counts = counts.tocsr()
    data = counts.data if sp.issparse(counts) else np.asarray(counts).ravel()
    assert np.allclose(data, np.round(data)), "counts 不是整数！"
    counts = counts.astype(np.int32)
    n_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    log(f"  n_counts: 中位 {np.median(n_counts):.0f}  范围 [{n_counts.min()}, {n_counts.max()}]")
    assert (n_counts > 0).all(), "存在零计数细胞"

    # ---------------- 3. var: symbol -> Ensembl ----------------
    g2i = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    symbols = b.var_names.astype(str).to_numpy()
    ensembl = np.array([g2i.get(s, "") for s in symbols], dtype=object)
    keep = ensembl != ""
    log(f"  基因符号可映射到 gc104M Ensembl: {keep.sum()} / {len(symbols)}")
    counts = counts[:, keep]
    new_var = pd.DataFrame(index=pd.Index(ensembl[keep].astype(str), name=None))
    new_var["feature_name"] = symbols[keep]
    # 去重（同一 Ensembl 可能对应多个 symbol）
    dup = new_var.index.duplicated()
    if dup.any():
        log(f"  去重 {int(dup.sum())} 个重复 Ensembl")
        counts = counts[:, ~dup]
        new_var = new_var[~dup]

    # ---------------- 4. obs ----------------
    obs = pd.DataFrame(index=pd.Index(b.obs_names.astype(str)))
    obs["n_counts"] = n_counts
    for c in KEEP_ATTRS:
        if c in b.obs.columns:
            obs[c] = b.obs[c].astype(str).values

    # ---------------- 5. 组装并保存 ----------------
    out = ad.AnnData(X=counts, obs=obs, var=new_var)
    if "X_umap" in b.obsm:
        out.obsm["X_umap"] = np.asarray(b.obsm["X_umap"])

    # VNN1 基线检查
    j = list(new_var.index).index(VNN1)
    vcol = np.asarray(counts[:, j].todense()).ravel() if sp.issparse(counts) \
        else np.asarray(counts[:, j]).ravel()
    log(f"  VNN1 原始计数: 表达细胞 {int((vcol > 0).sum())}/{len(vcol)} "
        f"({(vcol > 0).mean():.1%})  max={vcol.max():.0f}  总 counts={int(vcol.sum())}")

    path = DST_DIR / "GSE189175_malignant.h5ad"
    out.write_h5ad(path, compression="gzip")
    log(f"  已保存 -> {path}  ({out.shape[0]} 细胞 × {out.shape[1]} 基因)")
    log("完成")


if __name__ == "__main__":
    main()
