"""
重绘 VNN1 / FGB 两张 umap_displacement_arrows 图（英文，确定性防重叠布局 v3）。

左图：簇标签固定堆叠在右下角空白区，细引导线连到箭头尖端。
右图：簇均值颜色用 legend 标注（标准组件，绝不重叠），中位位移文字放角落。
数据来源：各 analysis 目录已有的 per_cell_shift.csv，无需重跑模型。
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")

JOBS = [
    ("VNN1", ROOT / "outputs" / "liver_isp_analysis", "umap_displacement_arrows_v3.png"),
    ("FGB", ROOT / "outputs" / "liver_isp_fgb_analysis", "umap_displacement_arrows_v3.png"),
]

SCALE = 30


def log(m=""):
    print(m, flush=True)


def draw(gene, ana_dir, out_name):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(ana_dir / "per_cell_shift.csv")
    df["leiden_1"] = df["leiden_1"].astype(str)  # 避免 np.where 混合类型把整数转成字符串
    xy_orig = df[["umap1", "umap2"]].values
    xy_pert = df[["umap1_pert", "umap2_pert"]].values
    med_disp = float(np.median(np.hypot(xy_pert[:, 0] - xy_orig[:, 0],
                                        xy_pert[:, 1] - xy_orig[:, 1])))
    dx = xy_pert[:, 0] - xy_orig[:, 0]
    dy = xy_pert[:, 1] - xy_orig[:, 1]

    top_cl = df["leiden_1"].value_counts().head(8).index
    df["cl_plot"] = np.where(df["leiden_1"].isin(top_cl), df["leiden_1"], "other")
    cats = sorted([c for c in top_cl], key=lambda x: -df["leiden_1"].value_counts()[x]) + ["other"]
    cmap = plt.get_cmap("tab20")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=140)

    # ---- 左：流形 + 簇均值箭头，标签堆叠在右下角 + 引导线 ----
    ax.scatter(xy_orig[:, 0], xy_orig[:, 1], s=7, color="#B4B2A9", alpha=0.4, linewidths=0)
    frac_ys = [0.10, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73]
    tips = []
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        if c == "other" or m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        coh = float(np.linalg.norm(mv) / np.mean(np.hypot(dx[m], dy[m])))
        med_d = float(np.median(np.hypot(dx[m], dy[m])))
        log(f"[{gene}] cluster {c}: n={m.sum()}  median disp {med_d:.3f}  coherence {coh:.2f}")
        x0, y0 = xy_orig[m].mean(axis=0)
        ex, ey = x0 + mv[0] * SCALE, y0 + mv[1] * SCALE
        ax.annotate("", xy=(ex, ey), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=2.8,
                                    mutation_scale=22))
        ax.scatter([x0], [y0], color="#A32D2D", s=34, zorder=5, edgecolor="k", linewidth=0.6)
        tips.append((ex, ey, f"c{c} (n={m.sum()})   med {med_d:.2f} · coh {coh:.2f}"))
    for k, (ex, ey, txt) in enumerate(tips):
        fy = frac_ys[k]
        ax.annotate("", xy=(ex, ey), xycoords="data",
                    xytext=(0.70, fy), textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-", color="#888780", lw=0.8,
                                    shrinkA=1, shrinkB=1))
        ax.text(0.715, fy, txt, transform=ax.transAxes, fontsize=9,
                color="#A32D2D", ha="left", va="center")
    ax.set_title(f"{gene} overexpression: cluster mean displacement (×{SCALE})", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # ---- 右：位移向量空间 + legend ----
    vmax = float(np.percentile(np.hypot(dx, dy), 99)) * 1.15
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        ax2.quiver(np.zeros(m.sum()), np.zeros(m.sum()), dx[m], dy[m],
                   angles="xy", scale_units="xy", scale=1, width=0.0035,
                   color=cmap(i % 20), alpha=0.35, linewidths=0)
    handles = []
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        if c == "other" or m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        ax2.annotate("", xy=mv, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=cmap(i % 20), lw=3,
                                     mutation_scale=20))
        handles.append(Line2D([0], [0], color=cmap(i % 20), lw=3,
                              label=f"c{c} mean (n={m.sum()})"))
    circ = plt.Circle((0, 0), med_disp, fill=False, color="#888780", ls="--", lw=1)
    ax2.add_patch(circ)
    handles.append(Line2D([0], [0], color="#888780", ls="--", lw=1,
                          label=f"median disp {med_disp:.2f}"))
    ax2.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92,
               borderpad=0.6, labelspacing=0.5)
    ax2.axhline(0, color="#D3D1C7", lw=0.6)
    ax2.axvline(0, color="#D3D1C7", lw=0.6)
    ax2.set_xlim(-vmax, vmax)
    ax2.set_ylim(-vmax, vmax)
    ax2.set_aspect("equal")
    ax2.set_title("Single-cell displacement vectors (original scale, from origin)",
                  fontsize=11)
    ax2.set_xlabel("dUMAP-1")
    ax2.set_ylabel("dUMAP-2")

    fig.tight_layout()
    out = ana_dir / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"[{gene}] saved -> {out}\n")


if __name__ == "__main__":
    for gene, ana_dir, out_name in JOBS:
        draw(gene, ana_dir, out_name)
    log("done")
