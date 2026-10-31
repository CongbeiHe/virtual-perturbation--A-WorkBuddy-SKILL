"""诊断 X 变换形式；验证 adata.raw 是否为原始整数计数。"""
import numpy as np
import scipy.sparse as sp
import anndata as ad

FILES = {
    "normal": r"E:\Geneformer\data\tokenized\_input\liver_normal.h5ad",
    "tumor":  r"E:\Geneformer\data\tokenized\_input\liver_tumor.h5ad",
}

def stats(M, name, nc=None):
    d = M.data if sp.issparse(M) else np.asarray(M).ravel()
    print(f"  [{name}] 非零 n={d.size}  min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.3f}")
    print(f"      整数占比={np.mean(np.abs(d-np.round(d))<1e-9):.4f}  "
          f"分位[50,99,99.9]%={np.percentile(d,[50,99,99.9]).round(2)}")

for tag, path in FILES.items():
    print("#" * 70)
    print(f"# {tag}")
    print("#" * 70)
    a = ad.read_h5ad(path)
    nc = a.obs["nCount_RNA"].values

    print("X:", end=""); stats(a.X, "X")
    if a.raw is not None:
        print("raw.X var 数:", a.raw.shape[1], "| var_names 前3:", list(a.raw.var_names[:3]))
        print("raw.X:", end=""); stats(a.raw.X, "raw")

    # 假设检验
    Xs = a.raw.X if a.raw is not None else a.X
    sums = np.asarray(Xs.sum(axis=1)).ravel()
    print(f"\nraw 每细胞总和 前5: {sums[:5].round(1)}")
    print(f"nCount_RNA 前5:    {nc[:5]}")
    ratio = sums / nc
    print(f"sum/nCount: 中位={np.median(ratio):.4f}  p10={np.percentile(ratio,10):.4f}  p90={np.percentile(ratio,90):.4f}")

    # log1p 假设: expm1(raw.X).sum() == ?
    ex = np.asarray(Xs.expm1().sum(axis=1)).ravel() if sp.issparse(Xs) else np.expm1(Xs).sum(axis=1)
    print(f"sum(expm1(raw.X)): 前5={ex[:5].round(1)}  中位={np.median(ex):.1f}")

    # X 层
    ex2 = np.asarray(a.X.expm1().sum(axis=1)).ravel() if sp.issparse(a.X) else np.expm1(a.X).sum(axis=1)
    print(f"sum(expm1(X)):     前5={ex2[:5].round(1)}  中位={np.median(ex2):.1f}")

    # 最大值细胞的整数性
    mx_cell = int(np.argmax(np.asarray(a.raw.X.max(axis=1).todense()).ravel())) if sp.issparse(a.raw.X) else 0
    row = a.raw.X[mx_cell].toarray().ravel() if sp.issparse(a.raw.X) else np.asarray(a.raw.X[mx_cell]).ravel()
    top = np.argsort(row)[::-1][:6]
    print(f"\nraw 计数最高细胞 #{mx_cell} (nCount={nc[mx_cell]:.0f}) top 基因:")
    for j in top:
        print(f"   raw={row[j]:12.4f}  -> round={round(row[j])}")
    print()
