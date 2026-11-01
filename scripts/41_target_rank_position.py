"""诊断 + 生成对照候选池：目标基因在 target⁺ 细胞中的 token rank 分布。

这一步决定「限制到 target⁺ 细胞」值不值得做 —— **先算 rank，再决定跑不跑**。

背景（SKILL.md §10.2）
--------------------
过表达的位移区间是 `[1, p-1]`（p = 目标基因在细胞内的 rank）。
  - p 小（如 192）：只位移前 ~191 个 token，**没有**覆盖整个 rank<500 头部。
    旧结果里不表达该基因的细胞走的是「整条序列全位移」的捷径，
    限制到 target⁺ 后这部分被拿掉 → 1-cos 会真的下降。值得做。
  - p ≳ 500：位移区间**几乎正好覆盖整个头部**，限制到 target⁺ 后机械签名
    原样保留，只是样本量塌掉。**不值得做**，应改用长度保持的 rank 编辑。

同时产出候选对照基因池（供 44 / 47 号脚本挑选）。候选池要开大
（`--top 300`）：默认 top-10 在两个数据集的交集往往不足 10 个，挑不出 20 个对照。

产出
----
  outputs/rank_match_<gene>_<ds>.csv   候选对照基因的匹配质量表
  outputs/rank_position_<ds>.csv        目标基因的 rank 分布汇总

用法
----
  python scripts/41_target_rank_position.py \
      --datasets myset --genes HIF3A --top 300
"""
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from _ranknull import (col, counts_matrix, find_gene, gene_symbols, out_dirs,
                       row_rank_stats, target_rank_in_cells, workdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--genes", nargs="+", required=True)
    ap.add_argument("--min-det-frac", type=float, default=0.5,
                    help="对照基因在 target+ 子集中的最低检出比例")
    ap.add_argument("--top", type=int, default=300,
                    help="每个目标基因保留的候选个数（挑 20 个对照需 >=300）")
    a = ap.parse_args()

    wd = workdir(a.workdir)
    DATA, OUT = out_dirs(wd)

    for ds in a.datasets:
        path = DATA / f"{ds}.h5ad"
        if not path.exists():
            print(f"[miss] {path}")
            continue
        ad = sc.read_h5ad(path)
        X = counts_matrix(ad)
        sym = gene_symbols(ad)
        n_cells, n_genes = X.shape
        print(f"\n########## {ds}  shape={X.shape}", flush=True)
        pos_rows = []

        for T in a.genes:
            j = find_gene(sym, T)
            if j < 0:
                print(f"  {T}: NOT FOUND")
                continue
            eid = str(ad.var_names[j]).split(".")[0]
            v = col(X, j)
            idx = np.where(v >= 1)[0]
            n_pos = len(idx)
            print(f"\n--- {T} ({eid})  target+ = {n_pos}/{n_cells} "
                  f"({100.0 * n_pos / n_cells:.2f}%)", flush=True)

            sub = X[idx]
            tr = target_rank_in_cells(sub, v, idx)
            det, med = row_rank_stats(sub, n_genes)

            t_med = float(np.median(tr))
            head_frac = float(np.mean(tr <= 500))
            print(f"  目标基因 rank: 中位={t_med:.0f}  "
                  f"IQR=[{np.percentile(tr, 25):.0f},{np.percentile(tr, 75):.0f}]  "
                  f"min={tr.min()}  max={tr.max()}")
            print(f"  p<=500 的细胞比例 = {head_frac:.1%}  "
                  f"(p>500 ⟹ 改动区间覆盖整个 rank<500 头部)")
            if t_med >= 500:
                print("  ⚠ rank 中位 >= 500：限制到 target+ 只砍样本量，"
                      "机械签名不会消失（见 SKILL §10.2）")
            pos_rows.append(dict(dataset=ds, gene=T, ensembl=eid, n_pos=n_pos,
                                 n_total=n_cells,
                                 pct=round(100.0 * n_pos / n_cells, 3),
                                 target_rank_median=t_med,
                                 target_rank_q25=float(np.percentile(tr, 25)),
                                 target_rank_q75=float(np.percentile(tr, 75)),
                                 target_rank_min=int(tr.min()),
                                 target_rank_max=int(tr.max()),
                                 frac_p_gt_500=round(head_frac, 4)))

            ok = (det >= max(30, a.min_det_frac * n_pos)) & np.isfinite(med)
            cand = np.where(ok)[0]
            cand = cand[cand != j]
            d = np.abs(med[cand] - t_med)
            order = cand[np.argsort(d)[:a.top]]
            rows = []
            for g in order:
                rows.append(dict(
                    gene=sym[g], ensembl=str(ad.var_names[g]).split(".")[0],
                    n_detected=int(det[g]),
                    det_frac=round(float(det[g]) / n_pos, 4),
                    rank_median=float(med[g]),
                    abs_rank_diff=float(abs(med[g] - t_med)),
                ))
            df = pd.DataFrame(rows)
            out = OUT / f"rank_match_{T}_{ds}.csv"
            df.to_csv(out, index=False)
            print(f"  候选对照基因（{len(df)} 个）-> {out}")

        if pos_rows:
            pd.DataFrame(pos_rows).to_csv(
                OUT / f"rank_position_{ds}.csv", index=False)


if __name__ == "__main__":
    main()
