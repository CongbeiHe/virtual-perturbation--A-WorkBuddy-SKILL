"""探测目标基因在各数据集中的检出率（counts>=1 的细胞数）。

用途：在决定「只对能检出该基因的细胞做虚拟过表达」之前，先看样本量够不够。
检出率过低（<1%）的基因，target⁺ 子集会小到没有统计效能，此时应改用
「修正扰动语义」（长度保持的 rank 编辑）而不是筛细胞 —— 见 SKILL.md §10.2。

产出
----
  outputs/detection_probe.csv   dataset, gene, n_pos, pct, max_umi, ...

用法
----
  python scripts/40_probe_target_detection.py \
      --datasets myset --genes HIF3A MAP3K6
"""
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from _ranknull import col, counts_matrix, find_gene, gene_symbols, out_dirs, workdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--genes", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="默认 outputs/detection_probe.csv")
    a = ap.parse_args()

    wd = workdir(a.workdir)
    _, OUT = out_dirs(wd)
    rows = []

    for ds in a.datasets:
        path = wd / "data" / f"{ds}.h5ad"
        if not path.exists():
            print(f"[miss] {path}")
            continue
        ad = sc.read_h5ad(path)
        X = counts_matrix(ad)
        sym = gene_symbols(ad)
        print(f"\n### {ds}: shape={ad.shape}")

        for g in a.genes:
            j = find_gene(sym, g)
            if j < 0:
                print(f"  {g}: NOT FOUND in var")
                rows.append(dict(dataset=ds, gene=g, n_total=ad.shape[0], n_pos=0,
                                 pct=0.0, max_umi=0, mean_umi_pos=0.0,
                                 single_umi_pct=0.0, ensembl="NA"))
                continue
            v = col(X, j)
            n = int((v > 0).sum())
            rows.append(dict(
                dataset=ds, gene=g, n_total=int(ad.shape[0]), n_pos=n,
                pct=round(100.0 * n / ad.shape[0], 3),
                max_umi=float(v.max()),
                mean_umi_pos=round(float(v[v > 0].mean()), 3) if n else 0.0,
                single_umi_pct=round(100.0 * (v == 1).sum() / n, 1) if n else 0.0,
                ensembl=str(ad.var_names[j]),
            ))
            print(f"  {g} ({ad.var_names[j]}): pos={n}/{ad.shape[0]} "
                  f"({100.0 * n / ad.shape[0]:.3f}%) max_umi={v.max():.0f} "
                  f"mean_umi(pos)={v[v > 0].mean() if n else 0:.2f} "
                  f"single_UMI={100.0 * (v == 1).sum() / n if n else 0:.1f}%")

    df = pd.DataFrame(rows)
    out = a.out or (OUT / "detection_probe.csv")
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
