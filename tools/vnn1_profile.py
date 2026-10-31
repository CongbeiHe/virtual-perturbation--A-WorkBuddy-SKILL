"""VNN1 阳性细胞的精确画像：按组织 x 细胞类型定位，供敲除实验选细胞。"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

H5AD = r"E:\Geneformer\data\tokenized\_input\normal_geneformer.h5ad"
VNN1 = "ENSG00000112299"

a = ad.read_h5ad(H5AD)
X = a.X.tocsc() if sp.issparse(a.X) else a.X
idx = int(np.where(np.asarray(a.var.index == VNN1) |
                   np.asarray(a.var["ensembl_id"].astype(str) == VNN1))[0][0])
col = np.asarray(X[:, idx].todense()).ravel() if sp.issparse(X) else np.asarray(X[:, idx]).ravel()

df = pd.DataFrame({
    "tissue": a.obs["tissue"].astype(str).values,
    "ct": a.obs["cell_type"].astype(str).values,
    "leiden": a.obs["leiden_1"].astype(str).values,
    "study": a.obs["harm_study"].astype(str).values,
    "n_counts": a.obs["n_counts"].values,
    "v": col,
})

print("=== VNN1 阳性细胞（>=1）按 组织 x 细胞类型 ===")
pos = df[df["v"] >= 1]
ct = pd.crosstab(pos["tissue"], pos["ct"])
print(ct.to_string())
print(f"\n阳性细胞合计: {len(pos)}")

print("\n=== VNN1 阳性细胞按 leiden_1 簇（组织分布）===")
t2 = pd.crosstab(pos["leiden"], pos["tissue"])
t2["total"] = t2.sum(axis=1)
t2["mean_expr"] = pos.groupby("leiden")["v"].mean().round(3)
t2["max_expr"] = pos.groupby("leiden")["v"].max()
print(t2.sort_values("total", ascending=False).head(12).to_string())

print("\n=== 高表达细胞（VNN1 >= 20）的归属 ===")
hi = df[df["v"] >= 20]
print(f"共 {len(hi)} 个高表达细胞")
print(pd.crosstab(hi["tissue"], hi["ct"]).to_string())
print()
print(hi.groupby(["tissue", "ct"])["v"].agg(["size", "mean", "max"]).round(1).to_string())

print("\n=== VNN1 在 leiden_1=16 / 28 / 23 / 19 簇的细胞类型 ===")
for cl in ["16", "28", "23", "19", "14", "1"]:
    s = df[df["leiden"] == cl]
    print(f"leiden {cl}: n={len(s)}, VNN1+={int((s['v']>=1).sum())}, "
          f"top 细胞类型={s['ct'].value_counts().head(2).to_dict()}")

print("\n=== 可用于敲除的候选池（VNN1 >= 阈值，且 n_counts 合理）===")
for t in [1, 2, 3, 5]:
    m = (df["v"] >= t) & (df["n_counts"] >= 500)
    print(f"  VNN1>={t} 且 n_counts>=500: {int(m.sum())} 细胞")
