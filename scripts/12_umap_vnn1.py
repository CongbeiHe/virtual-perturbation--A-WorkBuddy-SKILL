"""
VNN1 过表达后细胞状态的 UMAP 位移分析。

流程：
  1. 读取 tokenized tumor 数据，构造 原始 / 过表达后 两套 input_ids
     （复刻库内 overexpress_tokens 语义：VNN1 移到 <cls> 后第 1 位）
  2. 前向取最后一层 <cls> embedding（fp32，与库内 emb_layer=0 一致）
  3. 与库内 cell_embs pickle 交叉验证余弦相似度
  4. 原始 embedding 建 UMAP 流形，扰动后 embedding 经同一流形 transform
  5. 输出：流形着色图、按簇位移箭头图、位移强度箱线图、逐细胞坐标 CSV
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
DS_DIR = ROOT / "data" / "tokenized" / "liver_tumor.dataset"
RAW_DIR = ROOT / "outputs" / "liver_isp" / "tumor"
OUT_DIR = ROOT / "outputs" / "liver_isp_analysis"
BATCH = 8


def log(msg=""):
    print(msg, flush=True)


def load_token_dicts():
    code = ROOT / "code" / "geneformer"
    tok = pickle.load(open(code / "token_dictionary_gc104M.pkl", "rb"))
    return tok, tok["<cls>"], tok["<eos>"], tok["<pad>"], tok[VNN1]


def perturb_ids(ids, vnn1_tok, max_len):
    ids = list(ids)
    if vnn1_tok in ids:
        ids.remove(vnn1_tok)
    ids.insert(1, vnn1_tok)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [ids[-1]]
    return ids


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok, cls_id, eos_id, pad_id, vnn1_tok = load_token_dicts()
    from transformers import BertConfig, BertForMaskedLM

    import torch

    cfg = BertConfig.from_pretrained(str(MODEL_DIR))
    max_len = cfg.max_position_embeddings
    log(f"模型 max_position_embeddings = {max_len}")

    ds = load_from_disk(str(DS_DIR))
    all_ids = ds["input_ids"]
    n = len(ds)
    log(f"细胞数 {n}")

    leiden = [str(x) for x in ds["leiden_1"]]
    ctype = [str(x) for x in ds["cell_type"]]
    donor = [str(x) for x in ds["donor_id"]]

    # ---------------- 构造两套序列 ----------------
    orig_sorted = sorted(range(n), key=lambda i: -len(all_ids[i]))
    pert_ids = {i: perturb_ids(all_ids[i], vnn1_tok, max_len) for i in orig_sorted}

    # ---------------- 前向 ----------------
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
        e_pert = get_cls([pert_ids[i] for i in orig_sorted])
        emb_orig = np.zeros((n, e_orig.shape[1]), dtype=np.float32)
        emb_pert = np.zeros_like(emb_orig)
        for rank, i in enumerate(orig_sorted):
            emb_orig[i] = e_orig[rank]
            emb_pert[i] = e_pert[rank]
        np.savez_compressed(emb_file, orig=emb_orig, pert=emb_pert)
        log(f"embedding 已保存 -> {emb_file}")

    cos = np.sum(emb_orig * emb_pert, axis=1) / (
        np.linalg.norm(emb_orig, axis=1) * np.linalg.norm(emb_pert, axis=1) + 1e-12)

    # ---------------- 与库内结果交叉验证 ----------------
    pkls = list(RAW_DIR.glob("*_cell_embs_dict_*_raw.pickle"))
    if pkls:
        d = pickle.load(open(pkls[0], "rb"))
        ref = None
        for k, v in d.items():
            if "cell_emb" in k:
                ref = np.asarray(v, dtype=float)
                break
        if ref is not None:
            a, b = np.sort(ref), np.sort(cos)
            m = min(len(a), len(b))
            diff = float(np.max(np.abs(a[:m] - b[:m])))
            log(f"交叉验证: 库内 {len(ref)} 值 vs 本脚本 {len(cos)} 值, "
                f"排序后最大差 = {diff:.2e}  {'一致' if diff < 1e-3 else '⚠ 不一致'}")

    df = pd.DataFrame({
        "cell_idx": range(n), "leiden_1": leiden, "cell_type": ctype, "donor_id": donor,
        "cos_orig_pert": cos, "express_vnn1": [vnn1_tok in all_ids[i] for i in range(n)],
    })
    df.to_csv(OUT_DIR / "per_cell_shift.csv", index=False)
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
    df["umap1"] = xy_orig[:, 0]; df["umap2"] = xy_orig[:, 1]
    df["umap1_pert"] = xy_pert[:, 0]; df["umap2_pert"] = xy_pert[:, 1]
    df["umap_disp"] = disp
    df.to_csv(OUT_DIR / "per_cell_shift.csv", index=False)

    # ---------------- 画图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    top_cl = df["leiden_1"].value_counts().head(8).index
    df["cl_plot"] = np.where(df["leiden_1"].isin(top_cl), df["leiden_1"], "other")
    cats = sorted([c for c in top_cl], key=lambda x: -df["leiden_1"].value_counts()[x]) + ["other"]
    cmap = plt.get_cmap("tab20")

    # 图 1：流形 + 簇着色
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=140)
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        ax.scatter(df.loc[m, "umap1"], df.loc[m, "umap2"], s=8, alpha=0.8,
                   color=cmap(i % 20), label=f"cluster {c} (n={m.sum()})", linewidths=0)
    ax.set_title("HCC 恶性上皮细胞 · Geneformer <cls> embedding UMAP\n"
                 f"（{n} cells, 按 leiden_1 着色）", fontsize=11)
    ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.9)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_manifold.png", bbox_inches="tight")
    plt.close(fig)

    # 图 2：左 = 流形 + 簇均值位移箭头(×SCALE)；右 = 位移向量空间（原始尺度）
    SCALE = 30
    dx = xy_pert[:, 0] - xy_orig[:, 0]
    dy = xy_pert[:, 1] - xy_orig[:, 1]
    log("")
    log("簇位移方向凝聚度（1=完全同向）:")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=140)

    # ---- 左：流形 + 簇均值箭头 ----
    ax.scatter(xy_orig[:, 0], xy_orig[:, 1], s=7, color="#B4B2A9", alpha=0.4, linewidths=0)
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        if c == "other" or m.sum() < 5:
            continue
        mv = np.array([dx[m].mean(), dy[m].mean()])
        coh = float(np.linalg.norm(mv) / np.mean(np.hypot(dx[m], dy[m])))
        med_d = float(np.median(np.hypot(dx[m], dy[m])))
        log(f"  cluster {c}: n={m.sum()}  中位位移 {med_d:.3f}  凝聚度 {coh:.2f}  "
            f"平均方向 ({mv[0]:+.3f}, {mv[1]:+.3f})")
        x0, y0 = xy_orig[m].mean(axis=0)
        ax.annotate("", xy=(x0 + mv[0] * SCALE, y0 + mv[1] * SCALE), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=2.8,
                                    mutation_scale=22))
        ax.scatter([x0], [y0], color="#A32D2D", s=34, zorder=5, edgecolor="k", linewidth=0.6)
        ax.annotate(f"c{c} (n={m.sum()})\n中位 {med_d:.2f} · 凝聚 {coh:.2f}",
                    (x0 + mv[0] * SCALE, y0 + mv[1] * SCALE), fontsize=9, color="#A32D2D",
                    xytext=(6, 6), textcoords="offset points")
    ax.set_title(f"过表达 VNN1：簇平均位移（×{SCALE}）", fontsize=11)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # ---- 右：位移向量空间 ----
    vmax = float(np.percentile(np.hypot(dx, dy), 99)) * 1.15
    for i, c in enumerate(cats):
        m = (df["cl_plot"] == c).values
        ax2.quiver(np.zeros(m.sum()), np.zeros(m.sum()), dx[m], dy[m],
                   angles="xy", scale_units="xy", scale=1, width=0.0035,
                   color=cmap(i % 20), alpha=0.35, linewidths=0)
        if c != "other" and m.sum() >= 5:
            mv = np.array([dx[m].mean(), dy[m].mean()])
            ax2.annotate("", xy=mv, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=cmap(i % 20), lw=3,
                                         mutation_scale=20))
            ax2.annotate(f"c{c}", mv, fontsize=10, color=cmap(i % 20),
                         xytext=(4, 4), textcoords="offset points")
    circ = plt.Circle((0, 0), float(np.median(np.hypot(dx, dy))), fill=False,
                      color="#888780", ls="--", lw=1)
    ax2.add_patch(circ)
    ax2.annotate("中位位移 0.33", (0, float(np.median(np.hypot(dx, dy)))),
                 fontsize=8, color="#5F5E5A", xytext=(4, 2), textcoords="offset points")
    ax2.axhline(0, color="#D3D1C7", lw=0.6); ax2.axvline(0, color="#D3D1C7", lw=0.6)
    ax2.set_xlim(-vmax, vmax); ax2.set_ylim(-vmax, vmax)
    ax2.set_aspect("equal")
    ax2.set_title("单细胞位移向量（原始尺度，从原点出发）\n方向随机 → 凝聚度低，无全局定向迁移", fontsize=11)
    ax2.set_xlabel("ΔUMAP-1"); ax2.set_ylabel("ΔUMAP-2")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_displacement_arrows_v3.png", bbox_inches="tight")
    plt.close(fig)

    # 图 3：位移强度（按簇）
    order = (df.groupby("cl_plot")["umap_disp"].median().sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    data = [df.loc[df["cl_plot"] == c, "umap_disp"].values for c in order]
    axes[0].boxplot(data, tick_labels=order, showfliers=False)
    axes[0].set_title("UMAP 位移距离（按簇）")
    axes[0].set_ylabel("位移距离（UMAP 单位）")
    data2 = [df.loc[df["cl_plot"] == c, "cos_orig_pert"].values for c in order]
    axes[1].boxplot(data2, tick_labels=order, showfliers=False)
    axes[1].axhline(1.0, color="#888780", lw=0.8, ls="--")
    axes[1].set_title("embedding 余弦相似度（1=完全不变）")
    axes[1].set_ylabel("cos(原始, 扰动后)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "displacement_by_cluster.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\n图已保存到 {OUT_DIR}")
    log(f"总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
