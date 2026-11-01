"""rank-matched 对照零分布比较：目标基因信号是否超出机械位移。

两个检验
--------
【检验 A｜机制检验】同一次对照 run 内部，比较「表达该基因的细胞」与
  「不表达该基因的细胞」的 1-cos：
    - 表达   → 删原位 + 插回位置 1 = 长度不变，位移 [1, p-1]
    - 不表达 → 只 insert          = 长度 +1，位移全部 token
  若「不表达」组 1-cos 系统性更高，即证实机械伪迹的存在。

【检验 B｜零分布检验】在同一批细胞上，把目标基因的 1-cos 与 N 个 rank 匹配
  对照比较（均只取各自表达的细胞，保证同为长度保持的 rank 提升语义）：
    z       = (target - mean(controls)) / sd(controls)
    经验 p  = (k+1)/(n+1)，k = 对照中 >= 目标的个数
  补充：对「对照 1-cos ~ 对照 rank」做 OLS，用目标 rank 处的预测值作基准比残差，
  消掉对照之间残余的 rank 差异（20 个对照的 rank 不可能完全相等）。

判读（SKILL.md §10.6）
--------------------
- n=4 时经验 p 下限 0.20 → 不可能显著，只是提示性；要 >= 20 个对照。
- 凝聚度高只说明位移同向，**不代表基因特异性**；特异性只能靠本检验。

用法
----
  # 简单情形：对照与目标跑在同一批细胞上
  python scripts/46_null_comparison.py --target MAP3K6 \
      --datasets Atrial15k --controls-file outputs/ctl.txt

  # 对照跑在子样本上、需要按行号映射回切配对：
  python scripts/46_null_comparison.py --target HIF3A \
      --datasets Atrial15k Ventric15k \
      --controls-file outputs/hif3a20.txt \
      --null-dataset-pattern "{ds}_hif3ap_s2k" \
      --null-map-pattern "data/{ds}_hif3ap_s2k_idx.csv"

  # 多组实验一次跑完：
  python scripts/46_null_comparison.py --config outputs/exp.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from _ranknull import out_dirs, workdir


def _ranks_for(table_dir, target, dataset, genes):
    """取各基因在该数据集的 rank 中位（rank 校准检验用）。"""
    f = os.path.join(table_dir, f"rank_match_{target}_{dataset}.csv")
    if not os.path.exists(f):
        return None, None
    d = pd.read_csv(f)
    if "rank_median" not in d.columns or "gene" not in d.columns:
        return None, None
    rp = os.path.join(table_dir, f"rank_position_{dataset}.csv")
    tgt_rank = None
    if os.path.exists(rp):
        t = pd.read_csv(rp)
        if "gene" in t.columns:
            row = t[t["gene"].astype(str).str.upper() == target.upper()]
            for c in ("target_rank_median", "rank_median", "median"):
                if len(row) and c in row.columns:
                    tgt_rank = float(row.iloc[0][c])
                    break
    return dict(zip(d["gene"].astype(str), d["rank_median"])), tgt_rank


def load_shift(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if "cos_orig_pert" not in d.columns or "cell_idx" not in d.columns:
        return None
    d["one_minus_cos"] = 1.0 - d["cos_orig_pert"]
    return d


def build_experiments(a):
    if a.config:
        return json.load(open(a.config, encoding="utf-8"))
    controls = []
    if a.controls_file:
        controls = [g.strip() for g in open(a.controls_file, encoding="utf-8")
                    .read().splitlines() if g.strip()]
    controls += list(a.controls or [])
    if not controls:
        raise SystemExit("无对照基因：用 --controls / --controls-file / --config")
    tl = a.target.lower()
    return [dict(
        target=a.target,
        datasets=a.datasets,
        target_pattern=a.target_dataset_pattern or "{ds}_%sp" % tl,
        null_pattern=a.null_dataset_pattern or a.target_dataset_pattern
        or "{ds}_%sp" % tl,
        map_pattern=a.null_map_pattern,
        controls=controls,
    )]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--analysis-dir", default=None,
                    help="各 run 的父目录，默认 <workdir>/outputs")
    ap.add_argument("--config", default=None, help="JSON：多条实验配置")
    ap.add_argument("--target", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--controls", nargs="*", default=None)
    ap.add_argument("--controls-file", default=None)
    ap.add_argument("--target-dataset-pattern", default=None,
                    help="目标 run 数据集名模板，如 '{ds}_hif3ap'")
    ap.add_argument("--null-dataset-pattern", default=None,
                    help="对照 run 数据集名模板（跑在子样本上时不同）")
    ap.add_argument("--null-map-pattern", default=None,
                    help="子样本行号映射 csv 模板，如 'data/{ds}_hif3ap_s2k_idx.csv'")
    a = ap.parse_args()

    wd = workdir(a.workdir)
    _, OUT = out_dirs(wd)
    ANALYSIS = a.analysis_dir or OUT
    TABLE_DIR = OUT

    detail_rows, summary_rows = [], []
    for exp in build_experiments(a):
        T = exp["target"]
        for ds in exp["datasets"]:
            dsT = exp["target_pattern"].format(ds=ds, target=T.lower())
            dsNull = exp["null_pattern"].format(ds=ds, target=T.lower())
            tgt = load_shift(os.path.join(
                ANALYSIS, f"{dsT}_{T.lower()}_overexpress_analysis",
                "per_cell_shift.csv"))
            if tgt is None:
                print(f"[miss] 目标结果缺失: {dsT}_{T.lower()}_overexpress_analysis")
                continue
            tgt = tgt.set_index("cell_idx")

            mapping = None
            if exp.get("map_pattern"):
                mp = os.path.join(wd, exp["map_pattern"].format(
                    ds=ds, target=T.lower()))
                if os.path.exists(mp):
                    mapping = pd.read_csv(mp)["cell_idx"].to_numpy()
                else:
                    print(f"  [warn] 缺映射表 {mp}，改为不映射")

            print(f"\n=== {T} @ {ds}  (对照数据集={dsNull}, n_target={len(tgt)}) ===")
            controls = exp["controls"]

            # ---------- 检验 A：机制检验 ----------
            for c in controls:
                d = load_shift(os.path.join(
                    ANALYSIS, f"{dsNull}_{c.lower()}_overexpress_analysis",
                    "per_cell_shift.csv"))
                if d is None:
                    continue
                ecol = f"express_{c.lower()}"
                if ecol not in d.columns:
                    continue
                pos, neg = d[d[ecol]]["one_minus_cos"], d[~d[ecol]]["one_minus_cos"]
                if len(pos) < 10 or len(neg) < 10:
                    continue
                print(f"  机制 {c:<10} 表达 n={len(pos):<5} 1-cos={np.median(pos):.3e} | "
                      f"不表达 n={len(neg):<5} 1-cos={np.median(neg):.3e} | "
                      f"比值={np.median(neg) / np.median(pos):.3f}")

            # ---------- 检验 B：零分布 ----------
            ctl_meds, ctl_names = [], []
            for c in controls:
                d = load_shift(os.path.join(
                    ANALYSIS, f"{dsNull}_{c.lower()}_overexpress_analysis",
                    "per_cell_shift.csv"))
                if d is None:
                    continue
                ecol = f"express_{c.lower()}"
                if ecol not in d.columns:
                    continue
                sel = d[d[ecol]].copy()
                if len(sel) < 10:
                    continue
                sel["row"] = (mapping[sel["cell_idx"].to_numpy()]
                              if mapping is not None
                              else sel["cell_idx"].to_numpy())
                sel = sel.set_index("row")
                common = tgt.index.intersection(sel.index)
                if len(common) < 10:
                    continue
                t_med = float(np.median(tgt.loc[common, "one_minus_cos"]))
                c_med = float(np.median(sel.loc[common, "one_minus_cos"]))
                ctl_meds.append(c_med)
                ctl_names.append(c)
                detail_rows.append(dict(
                    target=T, dataset=ds, control=c, n_paired=len(common),
                    target_1cos_median=t_med, control_1cos_median=c_med,
                    ratio_target_over_control=t_med / c_med if c_med else np.nan,
                ))
                print(f"  零分布 {c:<10} 配对 n={len(common):<5} "
                      f"目标={t_med:.3e}  对照={c_med:.3e}  比值={t_med / c_med:.3f}")

            if not ctl_meds:
                continue
            arr = np.array(ctl_meds)
            t_all = float(np.median(tgt["one_minus_cos"]))
            sd = float(arr.std(ddof=1))
            z = (t_all - arr.mean()) / sd if sd > 0 else np.nan
            p_emp = (int(np.sum(arr >= t_all)) + 1) / (len(arr) + 1)
            row = dict(target=T, dataset=ds, n_target_cells=len(tgt),
                       target_1cos_median=t_all, ctl_mean=float(arr.mean()),
                       ctl_sd=sd, z_vs_null=z, emp_p=p_emp,
                       n_controls=len(arr))
            print(f"  >>> 目标 1-cos={t_all:.3e} | 对照 mean={arr.mean():.3e} "
                  f"sd={sd:.3e} | z={z:.2f} | 经验 p={p_emp:.3f}"
                  f"{'   ⚠ 对照 <20 个，p 下限 %.2f' % (1 / (len(arr) + 1)) if len(arr) < 20 else ''}")

            ranks, tgt_rank = _ranks_for(TABLE_DIR, T, ds, ctl_names)
            if ranks is not None and tgt_rank is not None and len(arr) >= 4:
                r = np.array([ranks.get(c, np.nan) for c in ctl_names], dtype=float)
                m = ~np.isnan(r)
                if m.sum() >= 4 and np.ptp(r[m]) > 0:
                    b, a0 = np.polyfit(r[m], arr[m], 1)
                    pred = a0 + b * tgt_rank
                    resid = arr[m] - (a0 + b * r[m])
                    t_resid = t_all - pred
                    sd_r = float(resid.std(ddof=1))
                    z_r = t_resid / sd_r if sd_r > 0 else np.nan
                    p_r = (int((resid >= t_resid).sum()) + 1) / (m.sum() + 1)
                    row.update(rank_slope=float(b), rank_pred_at_target=float(pred),
                               resid_z=z_r, resid_emp_p=p_r)
                    print(f"  >>> rank 校准：斜率={b:.2e}/rank  目标 rank={tgt_rank:.0f} "
                          f"预测={pred:.3e} | 残差 z={z_r:.2f} | 经验 p={p_r:.3f}")
            summary_rows.append(row)

    pd.DataFrame(detail_rows).to_csv(
        os.path.join(TABLE_DIR, "null_comparison_detail.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(TABLE_DIR, "null_comparison_summary.csv"), index=False)
    print(f"\n[saved] {TABLE_DIR}/null_comparison_detail.csv, "
          f"{TABLE_DIR}/null_comparison_summary.csv")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
