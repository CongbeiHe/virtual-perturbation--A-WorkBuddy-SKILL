"""
把 Ma et al. HCC 两份肝上皮数据集恢复成 Geneformer 可接受的格式。

数据现状：
  - X = log1p(CP10K)（已验证：每细胞 expm1(X) 行和 ≈ 10000）
  - adata.raw.X = 原始整数计数（已验证：100% 整数）
  - nCount_RNA 与 raw 行和不一致（不可信）
  - var 索引已是 Ensembl ID

处理：
  1. 用 raw.X 作为计数矩阵（int32）
  2. n_counts = raw.X 行和（真值）
  3. obs 只保留需要的元数据（全部转字符串）
  4. 保留 obsm["X_umap"] 供后续可视化参考
  5. 每个文件放入独立目录（tokenizer 按目录扫描，必须隔离）
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SRC = {
    "normal": r"E:\Geneformer\data\tokenized\_input\liver_normal.h5ad",
    "tumor": r"E:\Geneformer\data\tokenized\_input\liver_tumor.h5ad",
}
DST_ROOT = Path(r"E:\Geneformer\data\liver_counts")
VNN1 = "ENSG00000112299"
KEEP_ATTRS = ["cell_type", "author_cell_type", "leiden_0.8", "leiden_1",
              "donor_id", "disease", "tissue", "harm_sample.type"]


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for tag, path in SRC.items():
        print("=" * 70)
        print(f"[{tag}] 读取 {path}")
        a = ad.read_h5ad(path)

        counts = a.raw.X.copy() if a.raw is not None else a.X.copy()
        if sp.issparse(counts):
            counts = counts.tocsr()
        # 整数校验
        data = counts.data if sp.issparse(counts) else np.asarray(counts).ravel()
        assert np.allclose(data, np.round(data)), "raw.X 不是整数！"
        counts = counts.astype(np.int32) if sp.issparse(counts) else counts.astype(np.int32)

        n_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
        print(f"  cells={counts.shape[0]}  genes={counts.shape[1]}")
        print(f"  n_counts: 中位 {np.median(n_counts):.0f}  范围 [{n_counts.min()}, {n_counts.max()}]")
        assert (n_counts > 0).all(), "存在零计数细胞"

        # 检查 VNN1 原始计数
        var_ids = a.var.index.astype(str).to_numpy()
        j = int(np.where(var_ids == VNN1)[0][0])
        vcol = np.asarray(counts[:, j].todense()).ravel() if sp.issparse(counts) \
            else np.asarray(counts[:, j]).ravel()
        print(f"  VNN1 原始计数: 表达细胞 {int((vcol>0).sum())}/{len(vcol)} "
              f"({(vcol>0).mean():.1%})  max={vcol.max():.0f}  mean={vcol.mean():.3f}")

        # 组装新 AnnData
        obs = pd.DataFrame(index=a.obs_names)
        obs["n_counts"] = n_counts
        for c in KEEP_ATTRS:
            if c in a.obs.columns:
                obs[c] = a.obs[c].astype(str).values
        var = pd.DataFrame(index=a.var.index.astype(str))
        var["ensembl_id"] = var.index
        for c in ["feature_name", "feature_biotype"]:
            if c in a.var.columns:
                var[c] = a.var[c].values
        b = ad.AnnData(X=counts, obs=obs, var=var)
        if "X_umap" in a.obsm:
            b.obsm["X_umap"] = np.asarray(a.obsm["X_umap"])

        out_dir = DST_ROOT / tag
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"liver_{tag}.h5ad"
        b.write_h5ad(out_path, compression="gzip")
        print(f"  已保存 -> {out_path}")

    print("=" * 70)
    print("全部完成")


if __name__ == "__main__":
    main()
