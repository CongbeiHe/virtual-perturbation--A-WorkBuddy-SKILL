"""评估 VNN1+ vs VNN1- 分组 DE 的可行性：阈值敏感性、深度混杂、簇分布。

输出:
  outputs/nashhcc_vnn1_real_de/vnn1_group_feasibility.csv
  outputs/nashhcc_vnn1_real_de/vnn1_threshold_sensitivity.csv
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
gi = int(np.where(genes == "VNN1")[0][0])
indptr = f["layers/counts/indptr"][:]
cells = f["layers/counts/indices"][indptr[gi]:indptr[gi + 1]]
vals = f["layers/counts/data"][indptr[gi]:indptr[gi + 1]]


def cat(name):
    g = f[f"obs/{name}"]
    return dec(g["categories"][:])[g["codes"][:]]


sid, tis, cty = cat("SampleID"), cat("Tissue"), cat("cell_type")
mal = f["obs/malignant"][:]
ncounts = f["obs/n_counts"][:]
n = len(mal)

v = np.zeros(n, dtype=np.int64)
v[cells] = vals
sub = mal & (tis == "Tumor")

print("=== 1. VNN1 计数分布（肿瘤恶性细胞，仅阳性）===")
vc = v[sub]
dist = pd.Series(vc[vc > 0]).value_counts().sort_index()
print(dist.to_string())
print(f"阳性中 count==1 占 {(vc==1).sum()/max((vc>0).sum(),1):.1%}")

print("\n=== 2. 阈值敏感性 ===")
rows = []
for thr in [1, 2, 3, 4]:
    g1 = sub & (v >= thr)
    g0 = sub & (v < thr)
    rows.append({
        "threshold": thr,
        "n_pos": int(g1.sum()),
        "n_neg": int(g0.sum()),
        "pos_pct": float(g1.sum() / sub.sum()),
        "pos_median_ncounts": float(np.median(ncounts[g1])) if g1.any() else np.nan,
        "neg_median_ncounts": float(np.median(ncounts[g0])),
        "pos_pct_S1": float((g1 & (sid == "S1_hcc")).sum() / max((sub & (sid == "S1_hcc")).sum(), 1)),
        "pos_pct_S2": float((g1 & (sid == "S2_hcc")).sum() / max((sub & (sid == "S2_hcc")).sum(), 1)),
    })
ts = pd.DataFrame(rows)
print(ts.to_string(index=False))
ts.to_csv(OUT / "vnn1_threshold_sensitivity.csv", index=False)

print("\n=== 3. 测序深度混杂（threshold=1）===")
g1 = sub & (v >= 1)
print(f"阳性 n_counts 分位: {np.round(np.quantile(ncounts[g1], [0.25,0.5,0.75]),0)}")
print(f"阴性 n_counts 分位: {np.round(np.quantile(ncounts[sub & ~g1], [0.25,0.5,0.75]),0)}")
print("注：若阳性组深度显著更高，DE 会假阳性（文库大小混杂）")

print("\n=== 4. 与已有 S2 聚类的一致性 ===")
csv = Path(r"outputs/nashhcc_s2_clusters/s2_per_cell.csv")
if csv.exists():
    cl = pd.read_csv(csv)
    print("s2_per_cell 列:", list(cl.columns))
    # 用细胞条码匹配
    obs_names = dec(f["obs/_index"][:])
    pos_map = {obs_names[i]: int(v[i]) for i in np.where(sub)[0]}
    cl["vnn1"] = cl["cell"].map(pos_map) if "cell" in cl.columns else np.nan
    if cl["vnn1"].notna().any():
        print(cl.groupby("cluster")["vnn1"].agg(n="size", pos=lambda x: int((x > 0).sum()),
                                                pct=lambda x: float((x > 0).mean())).to_string())

print("\n=== 5. VNN1 在全肝细胞中的表现（可用于扩大样本的候选池）===")
hep = (cty == "Hepatocyte")
print(f"Hepatocyte 总数 {hep.sum()}  VNN1+ {((v > 0) & hep).sum()} ({((v > 0) & hep).mean():.2%})")
for t in sorted(set(tis)):
    m = hep & (tis == t)
    print(f"  {t:12s} n={m.sum():6d} 阳性 {(v[m]>0).sum():5d} ({(v[m]>0).mean():.2%})")
f.close()
