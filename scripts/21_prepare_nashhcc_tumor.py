"""
GSE189175 NASH-HCC：只抽取肿瘤组织中的恶性细胞（Tissue=Tumor & malignant=True），
转成 Geneformer 可接受格式。与 scripts/17 唯一区别是增加 Tissue 过滤。
"""

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(r"E:\Geneformer")
SRC = ROOT / "task" / "GSE189175" / "GSE189175_NASHHCC.h5ad"
DST_DIR = ROOT / "data" / "nashhcc_counts" / "tumor_mal"
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

    mask = (a.obs["malignant"].astype(bool) & (a.obs["Tissue"].astype(str) == "Tumor")).to_numpy()
    log(f"  Tumor & malignant 细胞 {mask.sum()} / {len(mask)}")
    b = a[mask].copy()
    log(f"  子集后: {b.shape[0]} 细胞  样本分布: {b.obs['SampleID'].value_counts().to_dict()}")

    counts = b.layers["counts"]
    if sp.issparse(counts):
        counts = counts.tocsr()
    data = counts.data if sp.issparse(counts) else np.asarray(counts).ravel()
    assert np.allclose(data, np.round(data)), "counts 不是整数！"
    counts = counts.astype(np.int32)
    n_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    log(f"  n_counts: 中位 {np.median(n_counts):.0f}  范围 [{n_counts.min()}, {n_counts.max()}]")
    assert (n_counts > 0).all(), "存在零计数细胞"

    g2i = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    symbols = b.var_names.astype(str).to_numpy()
    ensembl = np.array([g2i.get(s, "") for s in symbols], dtype=object)
    keep = ensembl != ""
    log(f"  基因符号可映射到 gc104M Ensembl: {keep.sum()} / {len(symbols)}")
    counts = counts[:, keep]
    new_var = pd.DataFrame(index=pd.Index(ensembl[keep].astype(str), name=None))
    new_var["feature_name"] = symbols[keep]
    dup = new_var.index.duplicated()
    if dup.any():
        log(f"  去重 {int(dup.sum())} 个重复 Ensembl")
        counts = counts[:, ~dup]
        new_var = new_var[~dup]

    obs = pd.DataFrame(index=pd.Index(b.obs_names.astype(str)))
    obs["n_counts"] = n_counts
    for c in KEEP_ATTRS:
        if c in b.obs.columns:
            obs[c] = b.obs[c].astype(str).values

    out = ad.AnnData(X=counts, obs=obs, var=new_var)
    if "X_umap" in b.obsm:
        out.obsm["X_umap"] = np.asarray(b.obsm["X_umap"])

    j = list(new_var.index).index(VNN1)
    vcol = np.asarray(counts[:, j].todense()).ravel() if sp.issparse(counts) \
        else np.asarray(counts[:, j]).ravel()
    log(f"  VNN1 原始计数: 表达细胞 {int((vcol > 0).sum())}/{len(vcol)} "
        f"({(vcol > 0).mean():.1%})  max={vcol.max():.0f}  总 counts={int(vcol.sum())}")

    path = DST_DIR / "GSE189175_tumor_mal.h5ad"
    out.write_h5ad(path, compression="gzip")
    log(f"  已保存 -> {path}  ({out.shape[0]} 细胞 × {out.shape[1]} 基因)")
    log("完成")


if __name__ == "__main__":
    main()
