"""把对照基因补足到 N 个（默认 20），用于让经验 p 有可能达到显著。

为什么必须 >= 20
----------------
经验 p = (k+1)/(n+1)。n=4 时下限 1/5 = 0.20 —— 无论目标基因多突出都下不去，
这是设计上限而非结果不显著。n=20 时下限 1/21 ≈ 0.048，才可能声称。
（Kendiukhov et al. 用了 61 个对照。）

挑选判据（按优先级）
--------------------
1. 同时落在**所有**给定数据集的候选表里 → 保证跨数据集可比
2. 同时存在于 `token_dictionary_gc104M.pkl` **和** `gene_name_id_dict_gc104M.pkl`
   （LINC02256 事故：有 ENSG 但无 token → run 全崩；只查前者会漏）
3. rank 中位尽量贴近目标基因：按 max(|Δrank|) 升序取
4. det_frac >= --min-det（比较脚本本身也会按 express_<gene> 列剔除未表达细胞，
   故 0.3 已有足够配对细胞）

入选基因的 rank 跨度**刻意横跨**目标的 rank（一部分比目标更靠后 → 机械位移更大
→ 检验偏保守，不易假阳性）。

前置：先跑 41 号脚本并把候选池开大（`--top 300`），否则交集不够。

产出
----
  outputs/null_controls_<TARGET><N>.csv
  outputs/<target><n>_genes.txt   纯基因名清单，供 45 号脚本直接读

用法
----
  python scripts/47_pick_extra_controls.py --target HIF3A \
      --datasets Atrial15k Ventric15k --existing CUX1 PLN FBXL7 RBFOX1 \
      --n-new 16
"""
import argparse

import pandas as pd

from _ranknull import code_dir, load_dicts, out_dirs, workdir


def target_ranks(OUT, target, datasets):
    """从 outputs/rank_position_<ds>.csv 读目标基因 rank 中位。"""
    out = {}
    for ds in datasets:
        f = OUT / f"rank_position_{ds}.csv"
        if not f.exists():
            raise SystemExit(f"缺 {f}（先跑 41 号脚本）")
        t = pd.read_csv(f)
        row = t[t["gene"].astype(str).str.upper() == target.upper()]
        if not len(row) or "target_rank_median" not in row.columns:
            raise SystemExit(f"{f} 中没有 {target} 的 target_rank_median")
        out[ds] = float(row.iloc[0]["target_rank_median"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--existing", nargs="*", default=[], help="已跑完的对照")
    ap.add_argument("--n-new", type=int, default=16)
    ap.add_argument("--max-rank-diff", type=float, default=60.0)
    ap.add_argument("--min-det", type=float, default=0.30)
    a = ap.parse_args()

    wd = workdir(a.workdir)
    cdir = code_dir(a.code_dir)
    _, OUT = out_dirs(wd)
    TOK, G2I = load_dicts(cdir)
    TR = target_ranks(OUT, a.target, a.datasets)

    frames = {}
    for ds in a.datasets:
        f = OUT / f"rank_match_{a.target}_{ds}.csv"
        if not f.exists():
            raise SystemExit(f"缺候选表 {f}（先跑 41 号脚本 --top 300）")
        d = pd.read_csv(f)
        d["ok"] = (d["ensembl"].isin(TOK.keys()) & d["gene"].isin(G2I.keys())
                   & ~d["gene"].isin(list(a.existing) + [a.target]))
        d["rank_diff"] = (d["rank_median"] - TR[ds]).abs()
        frames[ds] = d[d["ok"]].set_index("gene")

    common = frames[a.datasets[0]].index
    for ds in a.datasets[1:]:
        common = common.intersection(frames[ds].index)
    print(f"候选池交集（已过双字典校验）= {len(common)}")
    if len(common) < a.n_new:
        print(f"⚠ 可用 {len(common)} 个 < 需要的 {a.n_new}；"
              f"放宽 --max-rank-diff / --min-det，或把 41 号脚本的 --top 调更大")

    rows = []
    for g in common:
        rec = {"gene": g, "ensembl": frames[a.datasets[0]].loc[g, "ensembl"]}
        mx, mn = 0.0, 1.0
        for ds in a.datasets:
            r = frames[ds].loc[g]
            rec[f"rank_{ds}"] = r["rank_median"]
            rec[f"diff_{ds}"] = r["rank_diff"]
            rec[f"det_{ds}"] = r["det_frac"]
            mx = max(mx, r["rank_diff"])
            mn = min(mn, r["det_frac"])
        rec["max_rank_diff"], rec["min_det"] = mx, mn
        rows.append(rec)

    t = pd.DataFrame(rows)
    t = t[(t.max_rank_diff <= a.max_rank_diff) & (t.min_det >= a.min_det)]
    t = t.sort_values("max_rank_diff").reset_index(drop=True)
    print(f"门槛内（max|Δrank|<={a.max_rank_diff}, det>={a.min_det}）= {len(t)}")
    if len(t) < a.n_new:
        raise SystemExit(f"可用对照仅 {len(t)} 个，不足 {a.n_new}")

    new = t.head(a.n_new).copy()
    new["token"] = [TOK[e] for e in new["ensembl"]]
    print("\n新增对照（按 max|Δrank| 升序）:")
    print(new[["gene", "ensembl", "token", "max_rank_diff", "min_det"]]
          .to_string(index=False))

    allg = list(a.existing) + list(new["gene"])
    print(f"\n全部 {len(allg)} 个对照: {', '.join(allg)}")

    dst = OUT / f"null_controls_{a.target}{len(allg)}.csv"
    new.to_csv(dst, index=False)
    print(f"\n[saved] {dst}")

    lst = OUT / f"{a.target.lower()}{len(allg)}_genes.txt"
    lst.write_text("\n".join(allg) + "\n", encoding="utf-8")
    print(f"[saved] {lst}")


if __name__ == "__main__":
    main()
