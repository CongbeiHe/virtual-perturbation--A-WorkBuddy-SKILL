"""探查 GSE189175 原始 h5ad：VNN1 真实表达分布（用于后续 VNN1+ vs VNN1- DE）。

只读取 counts 中 VNN1 单列，避免全矩阵载入。
输出：outputs/nashhcc_vnn1_real_de/vnn1_expression_overview.csv
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

H5 = r"task/GSE189175/GSE189175_NASHHCC.h5ad"
OUT = Path(r"outputs/nashhcc_vnn1_real_de")
OUT.mkdir(parents=True, exist_ok=True)

f = h5py.File(H5, "r")


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


genes = dec(f["var/_index"][:])
print(f"基因数 {len(genes)}；示例 {genes[:5]}")
gi = int(np.where(genes == "VNN1")[0][0])
print(f"VNN1 var 索引 {gi}，ensembl {dec(f['var/ensembl_id'][gi:gi+1])}")

# counts 层以基因为主序存储（indptr 长度 = n_genes+1），可单列切片
indptr = f["layers/counts/indptr"][:]
s, e = int(indptr[gi]), int(indptr[gi + 1])
cells = f["layers/counts/indices"][s:e]
vals = f["layers/counts/data"][s:e]
print(f"VNN1 非零细胞 {e-s} 个；取值最大值 {vals.max()}；全整数={bool(np.all(vals == np.round(vals)))}")


def cat(name):
    g = f[f"obs/{name}"]
    cats = dec(g["categories"][:])
    return cats[g["codes"][:]]


sid = cat("SampleID")
tis = cat("Tissue")
cty = cat("cell_type")
mal = f["obs/malignant"][:]
n = len(mal)

v = np.zeros(n, dtype=np.int64)
v[cells] = vals
pos = v > 0

print(f"\n全数据 VNN1+ {pos.sum()}/{n} = {pos.mean():.2%}")
sub = mal & (tis == "Tumor")
print(f"肿瘤恶性细胞 {sub.sum()}，其中 VNN1+ {pos[sub].sum()} ({pos[sub].mean():.2%})")

rows = []
for s_ in sorted(set(sid[sub])):
    m = sub & (sid == s_)
    rows.append({
        "SampleID": s_, "n_cells": int(m.sum()),
        "n_vnn1_pos": int(pos[m].sum()),
        "pct_pos": float(pos[m].mean()),
        "median_count_pos": float(np.median(v[m][pos[m]])) if pos[m].any() else np.nan,
    })
df = pd.DataFrame(rows)
print("\n按样本（肿瘤恶性细胞）:")
print(df.to_string(index=False))

# 层次拆分，确认可用的比较策略
print("\nTissue x malignant:")
for t in sorted(set(tis)):
    for b in (False, True):
        m = (tis == t) & (mal == b)
        print(f"  {t:6s} malignant={b!s:5s} n={m.sum():6d} VNN1+ {pos[m].mean():6.2%}")

print("\ncell_type 分布（肿瘤恶性）:")
print(pd.Series(cty[sub]).value_counts().to_string())

df.to_csv(OUT / "vnn1_expression_overview.csv", index=False)
print(f"\n-> {OUT / 'vnn1_expression_overview.csv'}")
f.close()
