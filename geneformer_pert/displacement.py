"""步骤三：扰动前后 <cls> embedding 的 UMAP 位移分析。

移植自 scripts/23 + scripts/24 的置换基线，方法学约定：
  - 扰动序列构造 = 删原 token -> 插 <cls> 后位置 1 -> 超长截断保 <eos> 丢末位基因
  - fp32 前向、最后一层 hidden_states[:, 0, :]（<cls> 位置）
  - UMAP(n_neighbors=15, min_dist=0.5, cosine, random_state=42)
  - 凝聚度 = |均值位移向量| / 平均位移幅度
  - 随机基线 = 簇内重配对置换 N_PERM 次，报告均值/sd/95 分位
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BATCH, CODE_DIR, MODEL_DIR, N_PERM, PERTURB_INSERT_POS,
                     PERTURB_MODES, PERM_SEED, VRAM_LIMIT_FRAC, XVAL_TOL)


def log(m=""):
    print(m, flush=True)


def perturb_ids(ids, gene_tok, max_len, delete_only=False):
    """构造扰动后序列。

    - overexpress / activate / inhibit：删原 token -> 插到 <cls> 后位置 1
      （库内 perturb_emb_by_rank 语义）
    - delete（= knockout）：**只删除、不插回**（库内 perturber_utils.delete_indices 语义）

    ⚠ 2026-10-30 修复：此前无条件执行「删+插回」，导致 knockout 的 L1 位移/
    1-cos/凝聚度实际算的是**过表达**扰动。对照官方 raw pickle 的 cos：
    删+插回 与官方最大差 1.1e-3（超 XVAL_TOL），只删除则 1.8e-7（一致）。
    """
    ids = list(ids)
    if gene_tok in ids:
        ids.remove(gene_tok)
    if delete_only:
        return ids
    ids.insert(PERTURB_INSERT_POS, gene_tok)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [ids[-1]]
    return ids


def coherence(dx, dy):
    """方向凝聚度 = |均值向量| / 平均幅度。1=完全同向，~1/sqrt(n)=随机。"""
    mv = np.array([dx.mean(), dy.mean()])
    return float(np.linalg.norm(mv) / np.mean(np.hypot(dx, dy)))


def coherence_baseline(xy_orig, xy_pert, n_perm=N_PERM, seed=PERM_SEED):
    """簇内重配对置换基线：扰动坐标随机重新配对（保持幅度分布、破坏配对）。

    返回 dict(mean, sd, p95)。观测凝聚度 > p95 才能声称定向迁移。
    """
    rng = np.random.default_rng(seed)
    n = len(xy_orig)
    vals = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        v = xy_pert[p] - xy_orig
        vals[k] = coherence(v[:, 0], v[:, 1])
    return {"mean": float(vals.mean()), "sd": float(vals.std()),
            "p95": float(np.quantile(vals, 0.95))}


def group_stats(df, group_col, dx, dy, disp):
    """按分组列输出凝聚度表。"""
    rows = []
    for c in sorted(df[group_col].astype(str).unique()):
        m = (df[group_col].astype(str) == c).values
        if m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        rows.append({
            group_col: c, "n": int(m.sum()),
            "disp_median": float(np.median(disp[m])),
            "coherence": coherence(dx[m], dy[m]),
            "mean_dx": float(mv[0]), "mean_dy": float(mv[1]),
            "angle_deg": float(np.degrees(np.arctan2(mv[1], mv[0]))),
        })
    return pd.DataFrame(rows)


def run_displacement(dataset_name, target_gene, mode="overexpress",
                     dirs=None, batch=BATCH, make_plots=True, prev_analysis=None):
    """两遍前向 + UMAP 位移 + 凝聚度 + 交叉验证 + 图。返回输出路径 dict。"""
    import torch
    from datasets import load_from_disk
    from transformers import BertConfig, BertForMaskedLM

    from .config import default_dirs
    if dirs is None:
        dirs = default_dirs(dataset_name, target_gene, mode)
    out_dir = Path(dirs["analysis"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(dirs["isp_raw"])
    t0 = time.time()

    tok = pickle.load(open(CODE_DIR / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id, pad_id = tok["<cls>"], tok["<eos>"], tok["<pad>"]
    target_eid = target_gene if target_gene.startswith("ENSG") else None
    if target_eid is None:
        g2i = pickle.load(open(CODE_DIR / "gene_name_id_dict_gc104M.pkl", "rb"))
        target_eid = g2i[target_gene]
    tgt_tok = tok[target_eid]

    cfg = BertConfig.from_pretrained(str(MODEL_DIR))
    max_len = cfg.max_position_embeddings
    log(f"模型 max_position_embeddings = {max_len}")

    ds = load_from_disk(str(dirs["dataset"]))
    all_ids = ds["input_ids"]
    n = len(ds)
    log(f"细胞数 {n}")

    def col(name):
        return [str(x) for x in ds[name]] if name in ds.column_names else ["NA"] * n

    meta = {k: col(k) for k in ["cell_type", "SampleID", "Tissue", "Condition", "Patient"]}

    orig_sorted = sorted(range(n), key=lambda i: -len(all_ids[i]))
    delete_only = PERTURB_MODES.get(mode, mode) == "delete"
    pert = {i: perturb_ids(all_ids[i], tgt_tok, max_len, delete_only)
            for i in orig_sorted}
    log(f"序列构造：{'delete（只删除）' if delete_only else 'overexpress（删+插回 rank 1）'}")

    # ---------------- 前向（fp32, <cls> = hidden_states[-1][:,0,:]）
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from geneformer_safe import VRAMGuard, vram_report

    info = vram_report()
    log(f"设备: {info['device']}  总显存 {info['total_gb']}G  batch={batch}  精度 fp32")
    model = BertForMaskedLM.from_pretrained(str(MODEL_DIR))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    def get_cls(ids_list):
        outs = []
        with torch.no_grad():
            for s in range(0, len(ids_list), batch):
                chunk = ids_list[s: s + batch]
                L = max(len(x) for x in chunk)
                inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for r, seq in enumerate(chunk):
                    inp[r, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                    att[r, : len(seq)] = 1
                inp, att = inp.cuda(), att.cuda()
                with VRAMGuard(limit_frac=VRAM_LIMIT_FRAC, tag=f"fwd {s//batch}") as g:
                    hs = model(input_ids=inp, attention_mask=att,
                               output_hidden_states=True).hidden_states[-1]
                    outs.append(hs[:, 0, :].float().cpu().numpy())
                    g.check()
                del inp, att, hs
                torch.cuda.empty_cache()
                if (s // batch + 1) % 50 == 0:
                    log(f"  batch {s//batch + 1}/{(len(ids_list)+batch-1)//batch}")
        return np.concatenate(outs, axis=0)

    emb_file = out_dir / "cls_embs.npz"
    if emb_file.exists():
        log(f"[skip] 复用已有 embedding: {emb_file}")
        z = np.load(emb_file)
        emb_orig, emb_pert = z["orig"], z["pert"]
    else:
        log("前向：原始序列 ...")
        e_orig = get_cls([list(all_ids[i]) for i in orig_sorted])
        log(f"前向：{mode} {target_gene} 序列 ...")
        e_pert = get_cls([pert[i] for i in orig_sorted])
        emb_orig = np.zeros((n, e_orig.shape[1]), dtype=np.float32)
        emb_pert = np.zeros_like(emb_orig)
        for rank, i in enumerate(orig_sorted):
            emb_orig[i] = e_orig[rank]
            emb_pert[i] = e_pert[rank]
        np.savez_compressed(emb_file, orig=emb_orig, pert=emb_pert)
        log(f"embedding 已保存 -> {emb_file}")

    cos = np.sum(emb_orig * emb_pert, axis=1) / (
        np.linalg.norm(emb_orig, axis=1) * np.linalg.norm(emb_pert, axis=1) + 1e-12)

    # ---------------- 与库内 ISP 交叉验证（按长度降序对齐）
    pkls = list(raw_dir.glob("*_cell_embs_dict_*_raw.pickle"))
    xval = None
    if pkls:
        d = pickle.load(open(pkls[0], "rb"))
        for k, v in d.items():
            if "cell_emb" in k:
                ref = np.asarray(v, dtype=float)
                a, b2 = np.sort(ref), np.sort(cos)
                m = min(len(a), len(b2))
                xval = float(np.max(np.abs(a[:m] - b2[:m])))
                log(f"交叉验证: 库内 {len(ref)} vs 本脚本 {len(cos)}, "
                    f"排序后最大差 = {xval:.2e}  {'一致' if xval < XVAL_TOL else '⚠ 不一致'}")
                break

    df = pd.DataFrame({
        "cell_idx": range(n),
        **{k: v for k, v in meta.items()},
        "cos_orig_pert": cos,
        f"express_{target_gene.lower()}": [tgt_tok in all_ids[i] for i in range(n)],
    })
    log(f"逐细胞余弦: 中位 {np.median(cos):.5f}  1-cos 中位 {np.median(1-cos):.3e}")

    # ---------------- UMAP
    import umap
    from .config import UMAP_KW

    log("构建 UMAP 流形（原始 embedding）...")
    reducer = umap.UMAP(**UMAP_KW)
    xy_orig = reducer.fit_transform(emb_orig)
    log("投影扰动后 embedding 到同一流形 ...")
    xy_pert = reducer.transform(emb_pert)
    np.savez_compressed(out_dir / "umap_coords.npz", orig=xy_orig, pert=xy_pert)

    disp = np.linalg.norm(xy_pert - xy_orig, axis=1)
    dx, dy = xy_pert[:, 0] - xy_orig[:, 0], xy_pert[:, 1] - xy_orig[:, 1]
    df["umap1"], df["umap2"] = xy_orig[:, 0], xy_orig[:, 1]
    df["umap1_pert"], df["umap2_pert"] = xy_pert[:, 0], xy_pert[:, 1]
    df["umap_disp"] = disp
    per_cell_csv = out_dir / "per_cell_shift.csv"
    df.to_csv(per_cell_csv, index=False)
    log(f"UMAP 位移: 中位 {np.median(disp):.3f}  p90 {np.percentile(disp, 90):.3f}")

    # ---------------- 分组凝聚度 + 随机基线
    group_col = "SampleID" if df["SampleID"].nunique() > 1 else "cell_type"
    gs = group_stats(df, group_col, dx, dy, disp)
    gs.to_csv(out_dir / f"coherence_by_{group_col}.csv", index=False)
    log(f"\n按 {group_col} 的凝聚度（随机基线 p95≈{1/np.sqrt(max(n,1)):.3f} 量级）:")
    log(gs.to_string(index=False))

    base = coherence_baseline(xy_orig, xy_pert)
    log(f"\n全体重配对随机基线: mean {base['mean']:.3f}  sd {base['sd']:.3f}  p95 {base['p95']:.3f}")
    pd.Series(base).to_csv(out_dir / "coherence_baseline.csv")

    # ---------------- 稳健下游基因表（幅度参考，非显著性判据）
    gs_csv = Path(dirs["stats"]) / f"{dirs['tag']}_gene_shifts.csv"
    if gs_csv.exists():
        g = pd.read_csv(gs_csv)
        g["shift_down"] = 1 - g["Cosine_sim_mean"]
        robust = g[g["N_Detections"] >= 100].sort_values("shift_down").reset_index(drop=True)
        robust.to_csv(out_dir / "downstream_genes_robust.csv", index=False)
        log(f"\n幅度参考表（N_Detections>=100，非显著性判据）: {len(robust)} / {len(g)}")

    # ---------------- 图
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
        cats = list(df[group_col].astype(str).value_counts().index)
        for i, c in enumerate(cats):
            m = (df[group_col].astype(str) == c).values
            ax.scatter(df.loc[m, "umap1"], df.loc[m, "umap2"], s=7, alpha=0.8,
                       color=cmap(i % 10), linewidths=0, label=f"{c} (n={m.sum()})")
        ax.legend(fontsize=8, markerscale=2, framealpha=0.92, loc="upper left")
        ax.set_title(f"{dataset_name} · {mode} {target_gene} · <cls> UMAP ({n} cells)",
                     fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(out_dir / "umap_manifold.png", bbox_inches="tight")
        plt.close(fig)

        vmax = float(np.percentile(np.hypot(dx, dy), 99)) * 1.15
        fig, ax2 = plt.subplots(figsize=(7, 6.5), dpi=140)
        for i, c in enumerate(cats):
            m = (df[group_col].astype(str) == c).values
            ax2.quiver(np.zeros(m.sum()), np.zeros(m.sum()), dx[m], dy[m],
                       angles="xy", scale_units="xy", scale=1, width=0.0035,
                       color=cmap(i % 10), alpha=0.35, linewidths=0)
        for i, c in enumerate(cats):
            m = (df[group_col].astype(str) == c).values
            mv = np.array([dx[m].mean(), dy[m].mean()])
            ax2.annotate("", xy=mv, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=cmap(i % 10), lw=3,
                                         mutation_scale=20))
        circ = plt.Circle((0, 0), float(np.median(disp)), fill=False,
                          color="#888780", ls="--", lw=1)
        ax2.add_patch(circ)
        ax2.legend(handles=[Line2D([0], [0], color="#888780", ls="--", lw=1,
                                   label=f"median disp {np.median(disp):.2f}")],
                   loc="upper left", fontsize=8.5)
        ax2.axhline(0, color="#D3D1C7", lw=0.6); ax2.axvline(0, color="#D3D1C7", lw=0.6)
        ax2.set_xlim(-vmax, vmax); ax2.set_ylim(-vmax, vmax)
        ax2.set_aspect("equal")
        ax2.set_title(f"{mode} {target_gene}: single-cell displacement vectors", fontsize=11)
        ax2.set_xlabel("dUMAP-1"); ax2.set_ylabel("dUMAP-2")
        fig.tight_layout()
        fig.savefig(out_dir / "umap_displacement_arrows.png", bbox_inches="tight")
        plt.close(fig)

    log(f"\n总用时 {(time.time()-t0)/60:.1f} 分钟")
    return {"per_cell": per_cell_csv, "coherence": out_dir / f"coherence_by_{group_col}.csv",
            "baseline": base, "xval_max_diff": xval, "analysis_dir": out_dir,
            "group_col": group_col}
