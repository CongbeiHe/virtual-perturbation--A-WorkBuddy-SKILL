"""
步骤 4：对 Geneformer 细胞 embedding 做降维可视化，验证 embedding 的生物学意义。

如果 embedding 提取正确，UMAP 上不同细胞类型（T 细胞 / B 细胞 / 单核细胞 / NK）
应该形成清晰的分离簇 —— 这是判断「模型是否真的学到生物学表征」最直观的检验。

用法：
    python scripts/03_visualize_embeddings.py
    python scripts/03_visualize_embeddings.py --input outputs/pbmc3k_v2_104M.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 中文字体（缺失时静默回退）
for _f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser(description="Geneformer embedding 可视化")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "outputs" / "pbmc3k_v2_104M.csv"))
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "outputs" / "figures"))
    ap.add_argument("--prefix", default="pbmc3k_v2_104M")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"找不到 embedding 文件: {src}")

    df = pd.read_csv(src)
    print(f"读入: {src.name}  形状 {df.shape}")

    # 识别 embedding 维度列：整数列名
    emb_cols = [c for c in df.columns if str(c).isdigit()]
    if not emb_cols:
        raise SystemExit("未找到数值型 embedding 列")
    emb_cols = sorted(emb_cols, key=int)
    X = df[emb_cols].to_numpy(dtype=np.float32)
    print(f"embedding 矩阵: {X.shape}")

    # 标签列（可能是 cell_type 或 label）
    label_col = next((c for c in df.columns if c in ("cell_type", "label", "Celltype")), None)

    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    has_label = False
    if label_col:
        labels = df[label_col].astype(str).values
        n_types = len(set(labels))
        if n_types > 1:
            adata.obs["cell_type"] = pd.Categorical(labels)
            has_label = True
            vc = adata.obs["cell_type"].value_counts()
            print(f"\n细胞类型分布（{n_types} 类）:")
            for k, v in vc.items():
                print(f"  {k:<24} {v:>5}  ({v/len(df):.1%})")
        else:
            print(f"\n[注意] 标签列 '{label_col}' 只有 1 个取值（{labels[0]}），"
                  f"原始数据未带细胞类型注释，改用 Leiden 聚类验证 embedding 结构")

    # 标准降维流程
    # 注意：Geneformer embedding 是已上下文化的表征向量，不是原始计数，
    # 因此不做 normalize_total/normalize_per_cell（scanpy 1.12 起该函数已废弃、
    # 行为不稳定，会把 adata 改坏导致 PCA 报 n_features 异常）。
    sc.pp.pca(adata, n_comps=50, svd_solver="randomized")
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    print(f"\nPCA 方差解释率（前 5）: {np.round(adata.uns['pca']['variance_ratio'][:5], 4)}")

    # 无真实标签时，用 Leiden 分群考察 embedding 是否形成有意义的结构
    cluster_key = None
    if not has_label:
        sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph",
                     n_iterations=2, directed=False)
        cluster_key = "leiden"
        vc = adata.obs["leiden"].value_counts().sort_index()
        print(f"Leiden 聚类得到 {len(vc)} 个簇:")
        for k, v in vc.items():
            print(f"  cluster {k:<4} {v:>5}  ({v/len(df):.1%})")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    umap = adata.obsm["X_umap"]

    # 图 1：按细胞类型（或 Leiden 簇）着色
    color_key = "cell_type" if has_label else cluster_key
    if color_key:
        cats = list(adata.obs[color_key].astype("category").cat.categories)
        cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=140)
        for i, c in enumerate(cats):
            m = (adata.obs[color_key].astype(str) == str(c)).values
            ax.scatter(umap[m, 0], umap[m, 1], s=6, alpha=0.75,
                       color=cmap(i % 20), label=f"{c} (n={m.sum()})", linewidths=0)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        title = ("细胞类型" if has_label else "Leiden 聚类")
        ax.set_title(f"Geneformer V2-104M · PBMC3k 细胞 embedding（按{title}着色）\n"
                     f"{X.shape[0]} cells / {X.shape[1]}-dim", fontsize=11)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9, markerscale=2,
                  ncol=2 if len(cats) > 8 else 1)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p1 = out_dir / f"{args.prefix}_umap_by_{color_key}.png"
        fig.savefig(p1, bbox_inches="tight")
        plt.close(fig)
        print(f"\n已保存: {p1}")

    # 图 2：密度图（无标签时的兜底）
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    ax.scatter(umap[:, 0], umap[:, 1], s=4, alpha=0.5, c="#3B6FB6", linewidths=0)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Geneformer 细胞 embedding · UMAP", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = out_dir / f"{args.prefix}_umap.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {p2}")

    # 保存坐标，便于后续复用
    coords = pd.DataFrame(umap, columns=["UMAP1", "UMAP2"])
    coords.insert(0, "cell_index", range(len(coords)))
    if color_key:
        coords[color_key] = adata.obs[color_key].astype(str).values
    pc = out_dir / f"{args.prefix}_umap_coords.csv"
    coords.to_csv(pc, index=False)
    print(f"已保存: {pc}")


if __name__ == "__main__":
    main()
