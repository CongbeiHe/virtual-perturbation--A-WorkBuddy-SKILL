"""为 rank-matched 对照零分布抽固定大小的子样本（省 GPU）。

为什么需要
----------
目标基因的 target⁺ 子集可能有近万细胞，跑一次完整 L1 约 1 小时；
20 个对照 × 2 数据集全量跑 ≈ 数十小时。而 1-cos 是 CLS embedding 的余弦、
与 UMAP 无关，所以对照可以跑在一个固定子样本上，再从目标基因的全量结果里
按相同 cell_idx 切片做配对比较 —— 统计上足够（比的是同一批细胞中位数）。

对照基因很少（如 67~367 细胞）时不需要本脚本，直接全量跑。

产物
----
  data/<ds>_<suffix>.h5ad        子样本
  data/<ds>_<suffix>_idx.csv     子样本在源 h5ad 中的行号（配对回切用）

用法
----
  python scripts/43_make_null_subsample.py \
      --datasets Atrial15k_hif3ap Ventric15k_hif3ap --suffix s2k --n 2000
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd

from _ranknull import counts_matrix, workdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="源数据集名（通常就是 target+ 子集）")
    ap.add_argument("--suffix", default="s2k")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    wd = workdir(a.workdir)
    DATA = wd / "data"
    DATA.mkdir(parents=True, exist_ok=True)

    for ds in a.datasets:
        src = DATA / f"{ds}.h5ad"
        if not src.exists():
            print(f"[miss] {src}")
            continue
        h = ad.read_h5ad(src)
        n = h.shape[0]
        k = min(a.n, n)
        rng = np.random.default_rng(a.seed)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        X = counts_matrix(h)
        b = ad.AnnData(X=X[idx, :].copy(), obs=h.obs.iloc[idx].copy(),
                       var=h.var.copy())
        b.layers["counts"] = b.X.copy()
        dst = DATA / f"{ds}_{a.suffix}.h5ad"
        b.write(dst)
        pd.DataFrame({"cell_idx": idx}).to_csv(
            DATA / f"{ds}_{a.suffix}_idx.csv", index=False)
        print(f"{ds}: {n} -> {k} 细胞 (seed={a.seed}) -> {dst}")


if __name__ == "__main__":
    main()
