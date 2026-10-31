"""检查 liver_normal.h5ad 与 liver_tumor.h5ad 的结构与 VNN1 表达。"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

FILES = {
    "normal": r"E:\Geneformer\data\tokenized\_input\liver_normal.h5ad",
    "tumor":  r"E:\Geneformer\data\tokenized\_input\liver_tumor.h5ad",
}
VNN1 = "ENSG00000112299"

for tag, path in FILES.items():
    print("#" * 72)
    print(f"# {tag}: {path}")
    print("#" * 72)
    a = ad.read_h5ad(path)
    print(f"shape: {a.shape[0]} cells x {a.shape[1]} genes")
    print("obs columns:", list(a.obs.columns))
    print("var columns:", list(a.var.columns))
    print("X type:", type(a.X).__name__, "| dtype:", a.X.dtype)
    sub = a.X[:50, :300]
    d = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    print("X 样本 min/max:", d.min(), d.max(), "| 整数:", bool(np.allclose(d, np.round(d))))
    print()
    for c in a.obs.columns:
        vc = a.obs[c].astype(str).value_counts()
        head = ", ".join(f"{k}:{v}" for k, v in vc.head(6).items())
        print(f"  obs[{c}]  ({vc.shape[0]} 值)  {head}")
    print()
    print("var head:")
    print(a.var.head(5).to_string())
    print()

    # VNN1
    hits = np.where((a.var.index.astype(str) == VNN1) |
                    (a.var["ensembl_id"].astype(str) == VNN1))[0] if "ensembl_id" in a.var.columns \
          else np.where(a.var.index.astype(str) == VNN1)[0]
    if len(hits) == 0:
        print("!! VNN1 不在数据集中")
    else:
        idx = int(hits[0])
        X = a.X.tocsc() if sp.issparse(a.X) else a.X
        col = np.asarray(X[:, idx].todense()).ravel() if sp.issparse(X) else np.asarray(X[:, idx]).ravel()
        print(f"VNN1 索引 {idx}: 表达细胞 {int((col>0).sum())}/{len(col)} "
              f"({(col>0).mean():.2%})  max={col.max():.0f}  mean={col.mean():.3f}")
        for c in ["cell_type", "author_cell_type", "leiden", "leiden_1"]:
            if c in a.obs.columns:
                df = pd.DataFrame({"ct": a.obs[c].astype(str).values, "v": col})
                g = df.groupby("ct").agg(n=("v","size"), pos=("v", lambda s: int((s>0).sum())),
                                         mean=("v","mean"), mx=("v","max"))
                g["pos_frac"] = (g["pos"]/g["n"]).round(3)
                print(f"  --- by {c} (top10 by pos) ---")
                print(g.sort_values("pos", ascending=False).head(10).to_string())
    # n_counts
    for c in ["n_counts", "nCount_RNA"]:
        if c in a.obs.columns:
            v = a.obs[c].values
            print(f"{c}: 中位 {np.median(v):.0f}  分位[10,50,90]={np.percentile(v,[10,50,90]).round(0)}")
    print()
