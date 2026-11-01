"""
S2_hcc（3,424 个肿瘤恶性细胞）精细聚类 + 逐簇 VNN1 扰动状态变化分析。

流程：
  1. 载入已有 Geneformer <cls> embedding（outputs/nashhcc_tumor_vnn1_analysis/cls_embs.npz）
  2. scanpy: PCA(50) → neighbors(cosine) → Leiden（多分辨率比较后固定）
  3. 各簇 marker 基因（counts h5ad，Wilcoxon rank_genes_groups）
  4. 逐簇扰动状态变化：位移幅度/方向/凝聚度、1-cos、VNN1 表达比例
  5. 出图：UMAP 按簇着色 + 逐簇均值位移箭头
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")
EMB_DIR = ROOT / "outputs" / "nashhcc_tumor_vnn1_analysis"
DS_DIR = ROOT / "data" / "tokenized" / "nashhcc_tumor_mal.dataset"
COUNTS_H5 = ROOT / "data" / "nashhcc_counts" / "tumor_mal" / "GSE189175_tumor_mal.h5ad"
OUT_DIR = ROOT / "outputs" / "nashhcc_s2_clusters"
LEIDEN_RES = 0.5


def log(m=""):
    print(m, flush=True)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EMB_DIR / "per_cell_shift.csv")
    z = np.load(EMB_DIR / "cls_embs.npz")
    emb_all, xy_all = z["orig"], np.load(EMB_DIR / "umap_coords.npz")["orig"]
    is_s2 = (df["SampleID"] == "S2_hcc").to_numpy()
    s2_idx = np.where(is_s2)[0]
    n = len(s2_idx)
    log(f"S2_hcc 细胞: {n}")

    emb = emb_all[s2_idx]
    xy = xy_all[s2_idx]
    df_s2 = df.loc[s2_idx].reset_index(drop=True)

    # ---------------- 1. Leiden 聚类 ----------------
    import scanpy as sc
    import anndata as ad

    a = ad.AnnData(X=emb.astype(np.float32))
    a.obs_names = df_s2["cell_idx"].astype(str).values
    sc.pp.pca(a, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", metric="cosine")
    for res in [0.3, 0.5, 0.8, 1.2]:
        sc.tl.leiden(a, resolution=res, key_added=f"l{res}", flavor="igraph",
                     n_iterations=2, directed=False)
        sizes = a.obs[f"l{res}"].value_counts()
        log(f"res={res}: {len(sizes)} 簇  sizes={sizes.to_dict()}")
    a.obs["leiden"] = a.obs[f"l{LEIDEN_RES}"]
    cl = a.obs["leiden"].astype(str).to_numpy()
    df_s2["cluster"] = cl
    df_s2.to_csv(OUT_DIR / "s2_per_cell.csv", index=False)

    # ---------------- 2. marker 基因 ----------------
    log("\n计算各簇 marker 基因（Wilcoxon）...")
    b = ad.read_h5ad(COUNTS_H5)
    X = b.X
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.tocsr()[s2_idx]
    else:
        X = np.asarray(X)[s2_idx]
    ab = ad.AnnData(X=X.astype(np.float32),
                    obs=pd.DataFrame({"cluster": pd.Categorical(cl)}),
                    var=b.var.copy())
    sc.pp.normalize_total(ab, target_sum=1e4)
    sc.pp.log1p(ab)
    ab.var_names_make_unique()
    sc.tl.rank_genes_groups(ab, "cluster", method="wilcoxon", n_genes=25)
    mg = sc.get.rank_genes_groups_df(ab, group=None)
    mg.to_csv(OUT_DIR / "cluster_markers.csv", index=False)
    sym = b.var["feature_name"] if "feature_name" in b.var else None
    top5 = {}
    for g, sub in mg.groupby("group"):
        names = []
        for _, r in sub.head(6).iterrows():
            eid = r["names"]
            s = sym.get(eid, eid) if sym is not None else eid
            names.append(f"{s}({r['scores']:.1f})")
        top5[g] = names
        log(f"  c{g}: " + ", ".join(names))

    # ---------------- 3. 逐簇状态变化 ----------------
    dx_all = df_s2["umap1_pert"].to_numpy() - df_s2["umap1"].to_numpy()
    dy_all = df_s2["umap2_pert"].to_numpy() - df_s2["umap2"].to_numpy()
    disp = df_s2["umap_disp"].to_numpy()
    one_cos = 1 - df_s2["cos_orig_pert"].to_numpy()

    rows = []
    for c in sorted(set(cl), key=int):
        m = cl == c
        mv = np.array([dx_all[m].mean(), dy_all[m].mean()])
        coh = float(np.linalg.norm(mv) / np.mean(np.hypot(dx_all[m], dy_all[m])))
        ang = np.degrees(np.arctan2(mv[1], mv[0]))
        rows.append({
            "cluster": f"c{c}", "n_cells": int(m.sum()),
            "median_disp": float(np.median(disp[m])),
            "mean_disp": float(np.mean(disp[m])),
            "dir_x": float(mv[0]), "dir_y": float(mv[1]),
            "dir_angle_deg": float(ang), "coherence": coh,
            "one_cos_median": float(np.median(one_cos[m])),
            "vnn1_express_frac": float(df_s2.loc[m, "express_vnn1"].mean()),
            "top_markers": "; ".join(top5.get(c, [])[:5]),
        })
    stat = pd.DataFrame(rows)
    stat.to_csv(OUT_DIR / "cluster_state_change.csv", index=False)
    log("\n逐簇状态变化:")
    log(stat.drop(columns=["top_markers"]).to_string(index=False))

    # 全局随机基线（把簇标签打乱）
    rng = np.random.RandomState(0)
    null_coh = []
    for _ in range(200):
        perm = rng.permutation(n)
        fake = cl[perm]
        cohs = []
        for c in sorted(set(cl), key=int):
            m = fake == c
            if m.sum() < 10:
                continue
            mv = np.array([dx_all[m].mean(), dy_all[m].mean()])
            cohs.append(np.linalg.norm(mv) / np.mean(np.hypot(dx_all[m], dy_all[m])))
        null_coh.append(np.mean(cohs))
    log(f"\n随机标签基线: 平均凝聚度 {np.mean(null_coh):.3f} ± {np.std(null_coh):.3f} "
        f"(95% 分位 {np.quantile(null_coh, 0.95):.3f})")

    # ---------------- 4. 画图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cats = sorted(set(cl), key=int)
    cmap = plt.get_cmap("tab10")

    # 图 1：UMAP 按簇着色
    fig, ax = plt.subplots(figsize=(8, 6.8), dpi=140)
    for i, c in enumerate(cats):
        m = cl == c
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.75, linewidths=0,
                   color=cmap(i % 10), label=f"c{c} (n={m.sum()})")
    ax.legend(fontsize=8.5, markerscale=2.5, framealpha=0.92, loc="upper left",
              title=f"Leiden (res={LEIDEN_RES})")
    ax.set_title("S2_hcc tumor malignant cells · finer Leiden clusters\n"
                 f"({n} cells, Geneformer <cls> embedding)", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s2_umap_clusters.png", bbox_inches="tight")
    plt.close(fig)

    # 图 2：逐簇均值位移箭头（放大 S2 流形）+ 位移幅度箱线
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=140)
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#B4B2A9", alpha=0.35, linewidths=0)
    SCALE = 30
    for i, c in enumerate(cats):
        m = cl == c
        mv = np.array([dx_all[m].mean(), dy_all[m].mean()])
        x0, y0 = xy[m].mean(axis=0)
        ex, ey = x0 + mv[0] * SCALE, y0 + mv[1] * SCALE
        ax.annotate("", xy=(ex, ey), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(i % 10), lw=2.6,
                                    mutation_scale=20))
        ax.scatter([x0], [y0], color=cmap(i % 10), s=32, zorder=5,
                   edgecolor="k", linewidth=0.6)
        coh = stat.loc[stat["cluster"] == f"c{c}", "coherence"].iloc[0]
        med = stat.loc[stat["cluster"] == f"c{c}", "median_disp"].iloc[0]
        ax.text(x0, y0 - 0.55, f"c{c}\nmed {med:.2f} · coh {coh:.2f}",
                fontsize=8, ha="center", va="top", color=cmap(i % 10))
    ax.set_title(f"VNN1 overexpression: mean displacement per cluster (×{SCALE})", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    order = [f"c{c}" for c in cats]
    data = [disp[cl == c] for c in cats]
    ax2.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#E6F1FB"), medianprops=dict(color="#A32D2D"))
    ax2.set_title("UMAP displacement per cluster")
    ax2.set_ylabel("displacement (UMAP units)")
    ax2.set_xlabel("cluster")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s2_displacement_by_cluster.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\n图与表已保存到 {OUT_DIR}")
    log(f"总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
