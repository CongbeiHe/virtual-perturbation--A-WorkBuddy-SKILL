"""挑选 rank-matched 对照基因，构成过表达效应的经验零分布。

方法学依据（SKILL.md §10.8）
----------------------------
- Kendiukhov, *Comput Biol Chem* 2026;124(P1):109079 —— expression-matched
  empirical null（61 个对照基因），结果 0/21 通过 FDR，明确建议
  "expression-matched nulls as a minimum standard"。
- 在 Geneformer 中 token rank 就是细胞内表达降序位次，故 expression matching
  与 rank matching 在此表示下等价（本项目的迁移，非文献直引）。
- Geneformer 官方 `vs_null` 模式（`in_silico_perturber_stats.py` L465-521）
  也是拿目标基因 shift 对比随机基因集合做 Wilcoxon + BH FDR，代码现成。

关键约束
--------
1. 对照基因必须在同一批 target⁺ 细胞中**也有表达**，否则过表达退化成
   「只 insert 不删除」（长度 +1），与目标基因的「长度保持 rank 提升」不同构，
   比较失效。本脚本用 det_frac 卡这一点。
2. 对照必须同时在**两个 token 词典**里（见 `_ranknull.load_dicts` 注释）。
3. **个数 ≥ 20**：经验 p = (k+1)/(n+1)，n=4 时下限 0.20，无论多显著都下不去；
   n=20 时下限 1/21 ≈ 0.048。要扩到 20 个请用 47 号脚本。

产出
----
  outputs/null_controls.csv

用法
----
  python scripts/44_pick_null_controls.py \
      --targets HIF3A --datasets Atrial15k Ventric15k --n-controls 4
"""
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from _ranknull import (code_dir, col, counts_matrix, find_gene, gene_symbols,
                       load_dicts, out_dirs, row_rank_stats,
                       target_rank_in_cells, workdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n-controls", type=int, default=4)
    ap.add_argument("--min-det-frac", type=float, default=0.6)
    ap.add_argument("--max-diff", type=float, default=60.0,
                    help="两个数据集 rank 中位差上限（保证跨组织可比）")
    a = ap.parse_args()

    wd = workdir(a.workdir)
    cdir = code_dir(a.code_dir)
    DATA, OUT = out_dirs(wd)
    TOK, _ = load_dicts(cdir)

    cache = {}
    for ds in a.datasets:
        p = DATA / f"{ds}.h5ad"
        if not p.exists():
            raise SystemExit(f"缺数据: {p}")
        h = sc.read_h5ad(p)
        cache[ds] = (counts_matrix(h), gene_symbols(h),
                     np.asarray(h.var_names).astype(str))

    rows = []
    for T in a.targets:
        tgt_med, stats = {}, {}
        for ds in a.datasets:
            X, sym, _ = cache[ds]
            j = find_gene(sym, T)
            if j < 0:
                raise SystemExit(f"{ds} 中找不到 {T}")
            v = col(X, j)
            idx = np.where(v >= 1)[0]
            sub = X[idx]
            tgt_med[ds] = float(np.median(target_rank_in_cells(sub, v, idx)))
            det, med = row_rank_stats(sub, X.shape[1])
            stats[ds] = (det, med, len(idx), j)

        n_genes = cache[a.datasets[0]][0].shape[1]
        ok = np.ones(n_genes, dtype=bool)
        diff_sum = np.zeros(n_genes)
        for ds in a.datasets:
            det, med, n_pos, _ = stats[ds]
            ok &= (det >= a.min_det_frac * n_pos) & np.isfinite(med)
            diff_sum += np.abs(med - tgt_med[ds])

        # 排除目标基因自身与其它目标基因
        for T2 in a.targets:
            for ds in a.datasets:
                j2 = find_gene(cache[ds][1], T2)
                if j2 >= 0:
                    ok[j2] = False

        # ⚠ token 词典过滤：有 ENSG 但无 token 的基因会让 run 崩（LINC02256 事故）
        eids = cache[a.datasets[0]][2]
        in_tok = np.array([str(e).split(".")[0] in TOK for e in eids])
        n_drop = int((ok & ~in_tok).sum())
        if n_drop:
            print(f"  [{T}] 剔除无 token 的候选 {n_drop} 个")
        ok &= in_tok

        cand = np.where(ok)[0]
        cand = cand[np.argsort(diff_sum[cand])[:a.n_controls]]
        if not len(cand):
            print(f"  [{T}] 无可用对照：放宽 --min-det-frac / --max-diff，"
                  f"或先用 41 号脚本把候选池开大（--top 300）")
            continue

        for g in cand:
            for ds in a.datasets:
                det, med, n_pos, _ = stats[ds]
                rows.append(dict(
                    target=T, dataset=ds,
                    control=cache[ds][1][g],
                    control_ensembl=str(cache[ds][2][g]).split(".")[0],
                    n_pos=n_pos, n_paired=int(det[g]),
                    det_frac=round(float(det[g]) / n_pos, 4),
                    target_rank_median=round(tgt_med[ds], 1),
                    control_rank_median=round(float(med[g]), 1),
                    abs_rank_diff=round(abs(float(med[g]) - tgt_med[ds]), 1),
                ))

    df = pd.DataFrame(rows)
    out = OUT / "null_controls.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
