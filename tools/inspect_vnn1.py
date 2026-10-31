"""探查 VNN1 在 normal_geneformer.h5ad 中的表达分布，为敲除实验选细胞。"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

H5AD = r"E:\Geneformer\data\tokenized\_input\normal_geneformer.h5ad"
VNN1 = "ENSG00000112299"

print("读取数据集 ...", flush=True)
a = ad.read_h5ad(H5AD)
print(f"shape: {a.shape[0]} cells x {a.shape[1]} genes", flush=True)

# ---- 定位 VNN1 ----
var = a.var
mask = np.asarray(var.index == VNN1)
if not mask.any():
    mask = np.asarray(var["ensembl_id"].astype(str) == VNN1)
print("VNN1 在 var 中:", bool(mask.any()))
idx = int(np.where(mask)[0][0])
print("VNN1 索引:", idx, "| feature_name:", var.iloc[idx].get("feature_name"))

X = a.X.tocsc() if sp.issparse(a.X) else a.X
col = np.asarray(X[:, idx].todense()).ravel() if sp.issparse(X) else np.asarray(X[:, idx]).ravel()
print(f"\nVNN1 计数: 表达细胞 {int((col > 0).sum())} / {len(col)} "
      f"({(col > 0).mean():.2%}) | 最大 {col.max():.0f} | 均值 {col.mean():.3f}")

# ---- 按细胞类型统计 ----
ct = a.obs["cell_type"].astype(str)
df = pd.DataFrame({"ct": ct.values, "vnn1": col})
g = df.groupby("ct").agg(n=("vnn1", "size"), pos=("vnn1", lambda s: int((s > 0).sum())),
                         mean=("vnn1", "mean"), mx=("vnn1", "max"))
g["pos_frac"] = g["pos"] / g["n"]
g = g.sort_values("pos", ascending=False)
print("\n=== 按 cell_type 的 VNN1 表达（前 20）===")
print(g.head(20).to_string())

print("\n=== 按 leiden_1 的 VNN1 表达（前 15）===")
df2 = pd.DataFrame({"cl": a.obs["leiden_1"].astype(str).values, "vnn1": col})
g2 = df2.groupby("cl").agg(n=("vnn1", "size"), pos=("vnn1", lambda s: int((s > 0).sum())),
                           mean=("vnn1", "mean"))
g2["pos_frac"] = g2["pos"] / g2["n"]
print(g2.sort_values("pos", ascending=False).head(15).to_string())

# ---- 不同阈值下可选细胞数 ----
print("\n=== 不同表达阈值下的细胞数 ===")
for t in [1, 2, 3, 5, 8, 10]:
    m = col >= t
    print(f"  VNN1 >= {t:>2} : {int(m.sum()):>7} 细胞  ({m.mean():.2%})")

# ---- 总 counts 分布（用于后续筛选）----
nc = a.obs["n_counts"].values
print(f"\nn_counts: 中位 {np.median(nc):.0f} | 分位 [10%,50%,90%] = "
      f"{np.percentile(nc, [10, 50, 90]).round(0)}")

# ---- 整体 cell_type 分布 ----
print("\n=== 全部 cell_type 分布 ===")
print(a.obs["cell_type"].astype(str).value_counts().to_string())
