"""抽取「目标基因 counts>=1」的细胞子集（target⁺），供虚拟过表达使用。

为什么必须抽子集
----------------
过表达有两种 token 语义（SKILL.md §10.1）：
  - 基因**已表达** → 原位删除 + 插回位置 1 = 长度不变，rank p → 1（官方原生语义）
  - 基因**未表达** → 只 insert     = 长度 +1，整条序列全部位移（机械伪迹）
后者会把 1-cos 抬高约 8-11%，且不是生物学效应。抽 target⁺ 子集可消除该混杂。

产物
----
  data/<ds>_<tag>.h5ad               target⁺ 子集（保留 layers['counts'] 与 var）
  data/<ds>_<target>_expr.csv        父集全量细胞中该基因的 counts + 分组列

用法
----
  python scripts/42_make_pos_subset.py --target HIF3A --tag hif3ap \
      --datasets Atrial15k Ventric15k
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd

from _ranknull import col, counts_matrix, find_gene, gene_symbols, workdir

KEEP_COLS = ["n_counts", "n_genes", "cell_states", "cell_type", "SampleID",
             "Patient", "Tissue", "Condition", "disease", "sex", "donor_id",
             "sample", "type", "cell_source", "development_stage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tag", default=None, help="输出数据集后缀，默认 <target>p")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--expect-ensembl", default=None,
                    help="可选的 ENSG 校验；不符则报错退出")
    a = ap.parse_args()

    wd = workdir(a.workdir)
    DATA = wd / "data"
    DATA.mkdir(parents=True, exist_ok=True)
    T = a.target.upper()
    tag = a.tag or f"{T.lower()}p"

    for ds in a.datasets:
        src = DATA / f"{ds}.h5ad"
        if not src.exists():
            print(f"[miss] {src}")
            continue
        out = f"{ds}_{tag}"
        print(f"=== {ds} -> {out}", flush=True)
        h = ad.read_h5ad(src, backed="r")
        var = h.var
        sym = gene_symbols(h)
        j = find_gene(sym, T)
        if j < 0:
            raise SystemExit(f"{ds} 中找不到 {T}")
        eid = str(h.var_names[j]).split(".")[0]
        if a.expect_ensembl and eid != a.expect_ensembl:
            raise SystemExit(f"基因位点不符: {eid} != {a.expect_ensembl}")
        X = counts_matrix(h)
        expr = col(X, j)
        pos = expr >= 1
        print(f"  {T}+ {int(pos.sum())}/{len(expr)} ({pos.mean():.2%})  "
              f"单UMI {(expr[pos] == 1).mean():.1%}  max={expr.max()}", flush=True)

        keep = [c for c in KEEP_COLS if c in h.obs.columns]
        meta = h.obs[keep].copy()
        meta[T] = expr
        meta.to_csv(DATA / f"{ds}_{T.lower()}_expr.csv")
        print(f"  meta -> data/{ds}_{T.lower()}_expr.csv", flush=True)

        idx = np.where(pos)[0]
        sub = X[idx, :]
        b = ad.AnnData(X=sub.copy(), obs=h.obs.iloc[idx].copy(), var=var.copy())
        b.layers["counts"] = b.X.copy()
        b.obs[f"{T}_counts"] = expr[idx]
        dst = DATA / f"{out}.h5ad"
        b.write(dst)
        print(f"  子集 {b.shape} -> {dst}", flush=True)
        h.file.close()


if __name__ == "__main__":
    main()
