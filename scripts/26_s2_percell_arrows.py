"""
S2_hcc 逐细胞位移箭头图（参考 GRN perturbation 风格）：
  左面板: 每个细胞一支箭头（原 UMAP 位置 -> 扰动后位置），按 Leiden 亚簇着色
  右面板: 随机化对照（扰动后坐标在簇内随机重配对），方向凝聚度应坍缩
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
OUT_DIR = ROOT / "outputs" / "nashhcc_s2_clusters"

df = pd.read_csv(OUT_DIR / "s2_per_cell.csv")
n = len(df)
x0 = df["umap1"].to_numpy()
y0 = df["umap2"].to_numpy()
x1 = df["umap1_pert"].to_numpy()
y1 = df["umap2_pert"].to_numpy()
cl = df["cluster"].astype(str).to_numpy()
cats = sorted(set(cl), key=int)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

cmap = plt.get_cmap("tab10")
colors = {c: cmap(i % 10) for i, c in enumerate(cats)}


def coherence(x0, y0, x1, y1, mask):
    dx, dy = x1 - x0, y1 - y0
    mv = np.array([dx[mask].mean(), dy[mask].mean()])
    return float(np.linalg.norm(mv) / np.mean(np.hypot(dx[mask], dy[mask])))


def draw(ax, x0, y0, x1, y1, title):
    ax.scatter(x0, y0, s=4, color="#D3D1C7", alpha=0.35, linewidths=0, zorder=1)
    for c in cats:
        m = cl == c
        ax.quiver(x0[m], y0[m], x1[m] - x0[m], y1[m] - y0[m],
                  angles="xy", scale_units="xy", scale=1, width=0.0022,
                  color=colors[c], alpha=0.75, linewidths=0,
                  headwidth=3.2, headlength=3.8, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.3), dpi=140)

draw(ax, x0, y0, x1, y1, "VNN1 overexpression: per-cell displacement (observed)")
cohs = {c: coherence(x0, y0, x1, y1, cl == c) for c in cats}
leg = [plt.Line2D([0], [0], color=colors[c], lw=3,
                  label=f"c{c} (n={(cl==c).sum()}, coh {cohs[c]:.2f})") for c in cats]
ax.legend(handles=leg, fontsize=8.5, framealpha=0.92, loc="upper left")

# 随机化对照：扰动后坐标在簇内随机重配对（保持簇大小，破坏细胞-位移配对）
rng = np.random.RandomState(42)
xr, yr = x0.copy(), y0.copy()
for c in cats:
    m = cl == c
    idx = np.where(m)[0]
    perm = rng.permutation(idx)
    xr[idx], yr[idx] = x1[perm], y1[perm]
draw(ax2, x0, y0, xr, yr, "Randomized control: perturbed positions permuted within cluster")
cohs_r = {c: coherence(x0, y0, xr, yr, cl == c) for c in cats}
leg2 = [plt.Line2D([0], [0], color=colors[c], lw=3,
                   label=f"c{c} coh {cohs_r[c]:.2f}") for c in cats]
ax2.legend(handles=leg2, fontsize=8.5, framealpha=0.92, loc="upper left")

for ax_ in (ax, ax2):
    ax_.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "s2_percell_arrows.png", bbox_inches="tight")
plt.close(fig)

print("凝聚度对比（observed -> randomized）:")
for c in cats:
    print(f"  c{c}: {cohs[c]:.2f} -> {cohs_r[c]:.2f}")
print("已保存 -> outputs/nashhcc_s2_clusters/s2_percell_arrows.png")
