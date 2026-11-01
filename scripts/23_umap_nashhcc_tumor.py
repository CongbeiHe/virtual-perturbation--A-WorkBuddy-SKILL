"""
GSE189175 肿瘤组织恶性细胞 VNN1 过表达后的 UMAP 位移分析。
与 scripts/19 同逻辑，数据源/输出目录换成 tumor-only 子集。
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "code"))

from datasets import load_from_disk  # noqa: E402
from geneformer_safe import VRAMGuard, vram_report  # noqa: E402

VNN1 = "ENSG00000112299"
MODEL_DIR = ROOT / "models" / "V2-104M"
DS_DIR = ROOT / "data" / "tokenized" / "nashhcc_tumor_mal.dataset"
RAW_DIR = ROOT / "outputs" / "nashhcc_tumor_isp" / "raw"
STATS_DIR = ROOT / "outputs" / "nashhcc_tumor_isp_stats"
OUT_DIR = ROOT / "outputs" / "nashhcc_tumor_vnn1_analysis"
PREV_FULL_DIR = ROOT / "outputs" / "nashhcc_vnn1_analysis"   # 全 malignant（含癌旁）结果
BATCH = 8


def log(msg=""):
    print(msg, flush=True)


def perturb_ids(ids, gene_tok, max_len):
    ids = list(ids)
    if gene_tok in ids:
        ids.remove(gene_tok)
    ids.insert(1, gene_tok)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [ids[-1]]
    return ids


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    code = ROOT / "code" / "geneformer"
    tok = pickle.load(open(code / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id, pad_id = tok["<cls>"], tok["<eos>"], tok["<pad>"]
    vnn1_tok = tok[VNN1]

    from transformers import BertConfig, BertForMaskedLM
    import torch

    cfg = BertConfig.from_pretrained(str(MODEL_DIR))
    max_len = cfg.max_position_embeddings
    log(f"模型 max_position_embeddings = {max_len}")

    ds = load_from_disk(str(DS_DIR))
    all_ids = ds["input_ids"]
    n = len(ds)
    log(f"细胞数 {n}")

    def col(name):
        return [str(x) for x in ds[name]] if name in ds.column_names else ["NA"] * n

    ctype, sample, tissue, condition, patient = (col("cell_type"), col("SampleID"),
                                                col("Tissue"), col("Condition"),
                                                col("Patient"))

    orig_sorted = sorted(range(n), key=lambda i: -len(all_ids[i]))
    pert = {i: perturb_ids(all_ids[i], vnn1_tok, max_len) for i in orig_sorted}

    info = vram_report()
    log(f"设备: {info['device']}  总显存 {info['total_gb']}G  batch={BATCH}  精度 fp32")
    model = BertForMaskedLM.from_pretrained(str(MODEL_DIR))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    def get_cls(ids_list):
        outs = []
        with torch.no_grad():
            for s in range(0, len(ids_list), BATCH):
                chunk = ids_list[s : s + BATCH]
                L = max(len(x) for x in chunk)
                inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for r, seq in enumerate(chunk):
                    inp[r, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                    att[r, : len(seq)] = 1
                inp, att = inp.cuda(), att.cuda()
                with VRAMGuard(limit_frac=0.85, tag=f"fwd {s//BATCH}") as g:
                    hs = model(input_ids=inp, attention_mask=att,
                               output_hidden_states=True).hidden_states[-1]
                    outs.append(hs[:, 0, :].float().cpu().numpy())
                    g.check()
                del inp, att, hs
                torch.cuda.empty_cache()
                if (s // BATCH + 1) % 50 == 0:
                    log(f"  batch {s//BATCH + 1}/{(len(ids_list)+BATCH-1)//BATCH}")
        return np.concatenate(outs, axis=0)

    emb_file = OUT_DIR / "cls_embs.npz"
    if emb_file.exists():
        log(f"[skip] 复用已有 embedding: {emb_file}")
        z = np.load(emb_file)
        emb_orig, emb_pert = z["orig"], z["pert"]
    else:
        log("前向：原始序列 ...")
        e_orig = get_cls([list(all_ids[i]) for i in orig_sorted])
        log("前向：过表达 VNN1 序列 ...")
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

    # 与库内 ISP 结果交叉验证
    pkls = list(RAW_DIR.glob("*_cell_embs_dict_*_raw.pickle"))
    if pkls:
        d = pickle.load(open(pkls[0], "rb"))
        for k, v in d.items():
            if "cell_emb" in k:
                ref = np.asarray(v, dtype=float)
                a, b = np.sort(ref), np.sort(cos)
                m = min(len(a), len(b))
                diff = float(np.max(np.abs(a[:m] - b[:m])))
                log(f"交叉验证: 库内 {len(ref)} 值 vs 本脚本 {len(cos)} 值, "
                    f"排序后最大差 = {diff:.2e}  {'一致' if diff < 1e-3 else '⚠ 不一致'}")
                break

    df = pd.DataFrame({
        "cell_idx": range(n), "cell_type": ctype, "SampleID": sample, "Tissue": tissue,
        "Condition": condition, "Patient": patient,
        "cos_orig_pert": cos, "express_vnn1": [vnn1_tok in all_ids[i] for i in range(n)],
    })
    log(f"逐细胞余弦: 中位 {np.median(cos):.5f}  分位[10,90]={np.percentile(cos,[10,90]).round(5)}")

    # ---------------- UMAP ----------------
    import umap

    log("构建 UMAP 流形（原始 embedding）...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.5, metric="cosine",
                        random_state=42, n_jobs=1)
    xy_orig = reducer.fit_transform(emb_orig)
    log("投影扰动后 embedding 到同一流形 ...")
    xy_pert = reducer.transform(emb_pert)
    np.savez_compressed(OUT_DIR / "umap_coords.npz", orig=xy_orig, pert=xy_pert)

    disp = np.linalg.norm(xy_pert - xy_orig, axis=1)
    df["umap1"], df["umap2"] = xy_orig[:, 0], xy_orig[:, 1]
    df["umap1_pert"], df["umap2_pert"] = xy_pert[:, 0], xy_pert[:, 1]
    df["umap_disp"] = disp
    df.to_csv(OUT_DIR / "per_cell_shift.csv", index=False)
    med_disp = float(np.median(disp))
    log(f"UMAP 位移: 中位 {med_disp:.3f}  p90 {np.percentile(disp, 90):.3f}")

    # ---------------- 稳健下游基因表 ----------------
    gs = STATS_DIR / "vnn1_oe_gene_shifts.csv"
    if gs.exists():
        g = pd.read_csv(gs)
        g["shift_down"] = 1 - g["Cosine_sim_mean"]
        robust = g[g["N_Detections"] >= 100].sort_values("shift_down").reset_index(drop=True)
        robust.to_csv(OUT_DIR / "downstream_genes_robust.csv", index=False)
        log(f"\n稳健下游基因（N_Detections>=100）: {len(robust)} / {len(g)}")
        log(robust.head(15)[["Affected_gene_name", "Affected_Ensembl_ID",
                             "Cosine_sim_mean", "N_Detections", "shift_down"]]
            .to_string(index=False))

    # ---------------- 画图（英文、确定性防重叠布局） ----------------
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for key, fname, title in [("SampleID", "umap_manifold_sample.png", "by sample"),
                              ("Tissue", "umap_manifold_tissue.png", "by tissue")]:
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
        cats = list(pd.Series(df[key]).value_counts().index)
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(cats):
            m = (df[key] == c).values
            ax.scatter(df.loc[m, "umap1"], df.loc[m, "umap2"], s=7, alpha=0.8,
                       color=cmap(i % 10), linewidths=0, label=f"{c} (n={m.sum()})")
        ax.legend(fontsize=8, markerscale=2, framealpha=0.92, loc="upper left")
        ax.set_title(f"NASH-HCC tumor malignant cells · Geneformer <cls> UMAP\n({n} cells, {title})",
                     fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        plt.close(fig)

    top_cl = df["SampleID"].value_counts().head(8).index
    df["grp"] = np.where(df["SampleID"].isin(top_cl), df["SampleID"].astype(str), "other")
    cats = sorted([c for c in top_cl], key=lambda x: -df["SampleID"].value_counts()[x]) + ["other"]
    cmap = plt.get_cmap("tab10")
    dx = xy_pert[:, 0] - xy_orig[:, 0]
    dy = xy_pert[:, 1] - xy_orig[:, 1]
    SCALE = 30

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=140)
    ax.scatter(xy_orig[:, 0], xy_orig[:, 1], s=6, color="#B4B2A9", alpha=0.4, linewidths=0)
    frac_ys = [0.10, 0.19, 0.28, 0.37, 0.46, 0.55]
    tips = []
    log("")
    log("样本均值位移方向凝聚度（1=完全同向）:")
    for k, c in enumerate(cats):
        m = (df["grp"] == c).values
        if c == "other" or m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        coh = float(np.linalg.norm(mv) / np.mean(np.hypot(dx[m], dy[m])))
        med_d = float(np.median(np.hypot(dx[m], dy[m])))
        log(f"  {c}: n={m.sum()}  中位位移 {med_d:.3f}  凝聚度 {coh:.2f}  "
            f"平均方向 ({mv[0]:+.3f}, {mv[1]:+.3f})")
        x0, y0 = xy_orig[m].mean(axis=0)
        ex, ey = x0 + mv[0] * SCALE, y0 + mv[1] * SCALE
        ax.annotate("", xy=(ex, ey), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=2.8,
                                    mutation_scale=22))
        ax.scatter([x0], [y0], color="#A32D2D", s=34, zorder=5, edgecolor="k", linewidth=0.6)
        tips.append((ex, ey, f"{c} (n={m.sum()})   med {med_d:.2f} · coh {coh:.2f}"))
    for k, (ex, ey, txt) in enumerate(tips):
        fy = frac_ys[min(k, len(frac_ys) - 1)]
        ax.annotate("", xy=(ex, ey), xycoords="data",
                    xytext=(0.66, fy), textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-", color="#888780", lw=0.8,
                                    shrinkA=1, shrinkB=1))
        ax.text(0.675, fy, txt, transform=ax.transAxes, fontsize=8.5,
                color="#A32D2D", ha="left", va="center")
    ax.set_title(f"VNN1 overexpression: mean displacement per sample (×{SCALE})", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    vmax = float(np.percentile(np.hypot(dx, dy), 99)) * 1.15
    handles = []
    for i, c in enumerate(cats):
        m = (df["grp"] == c).values
        ax2.quiver(np.zeros(m.sum()), np.zeros(m.sum()), dx[m], dy[m],
                   angles="xy", scale_units="xy", scale=1, width=0.0035,
                   color=cmap(i % 10), alpha=0.35, linewidths=0)
        if c == "other" or m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        ax2.annotate("", xy=mv, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=cmap(i % 10), lw=3,
                                     mutation_scale=20))
        handles.append(Line2D([0], [0], color=cmap(i % 10), lw=3,
                              label=f"{c} mean (n={m.sum()})"))
    circ = plt.Circle((0, 0), med_disp, fill=False, color="#888780", ls="--", lw=1)
    ax2.add_patch(circ)
    handles.append(Line2D([0], [0], color="#888780", ls="--", lw=1,
                          label=f"median disp {med_disp:.2f}"))
    ax2.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.92,
               borderpad=0.6, labelspacing=0.5)
    ax2.axhline(0, color="#D3D1C7", lw=0.6)
    ax2.axvline(0, color="#D3D1C7", lw=0.6)
    ax2.set_xlim(-vmax, vmax)
    ax2.set_ylim(-vmax, vmax)
    ax2.set_aspect("equal")
    ax2.set_title("Single-cell displacement vectors (original scale, from origin)", fontsize=11)
    ax2.set_xlabel("dUMAP-1")
    ax2.set_ylabel("dUMAP-2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_displacement_arrows_v3.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    order = df.groupby("grp")["umap_disp"].median().sort_values(ascending=False).index
    data = [df.loc[df["grp"] == c, "umap_disp"].values for c in order]
    axes[0].boxplot(data, tick_labels=order, showfliers=False)
    axes[0].set_title("UMAP displacement distance (by sample)")
    axes[0].set_ylabel("displacement (UMAP units)")
    data2 = [df.loc[df["grp"] == c, "cos_orig_pert"].values for c in order]
    axes[1].boxplot(data2, tick_labels=order, showfliers=False)
    axes[1].axhline(1.0, color="#888780", lw=0.8, ls="--")
    axes[1].set_title("embedding cosine similarity (1 = unchanged)")
    axes[1].set_ylabel("cos(orig, perturbed)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "displacement_by_sample.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- 与全 malignant 版结果对比 ----------------
    log("")
    log("=" * 60)
    log("与全 malignant 版（含癌旁 151 细胞）对比:")
    rows = [{"subset": "tumor-only malignant", "n_cells": n,
             "1-cos median": float(np.median(1 - cos)), "UMAP disp median": med_disp}]
    prev_csv = PREV_FULL_DIR / "per_cell_shift.csv"
    if prev_csv.exists():
        pv = pd.read_csv(prev_csv)
        rows.append({"subset": "all malignant (incl. adjacent)",
                     "n_cells": len(pv),
                     "1-cos median": float(np.median(1 - pv["cos_orig_pert"])),
                     "UMAP disp median": float(np.median(pv["umap_disp"]))})
    cmp_df = pd.DataFrame(rows)
    log(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT_DIR / "comparison_subsets.csv", index=False)

    log(f"\n图与表已保存到 {OUT_DIR}")
    log(f"总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
