"""
S2_hcc 各 Leiden 亚簇的 VNN1 扰动基因级响应。

原理：库内 gene_embs pickle 键 (VNN1_token, gene_token) -> 余弦列表，
列表 = 按「序列长度降序的细胞处理顺序」且仅含含该基因的细胞（已验证：
长度与库内 N_Detections 一致、均值与库内 Cosine_sim_mean 一致到 1e-6）。
将每个列表位置映射回细胞 -> 亚簇标签，即可得逐簇基因位移，无需重跑模型。

输出：
  cluster_gene_shifts_long.csv      gene × cluster 长表
  cluster_top_genes.csv             每簇位移最大的 30 个基因（簇内 N>=20）
  cluster_specific_genes.csv        簇特异性基因（该簇位移 - 其余簇合并位移）
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")
DS_DIR = ROOT / "data" / "tokenized" / "nashhcc_tumor_mal.dataset"
PKL = ROOT / "outputs" / "nashhcc_tumor_isp" / "raw" / \
    "in_silico_overexpress_vnn1_oe_gene_embs_dict_[4005]_raw.pickle"
CLUSTERS_CSV = ROOT / "outputs" / "nashhcc_s2_clusters" / "s2_per_cell.csv"
OUT_DIR = ROOT / "outputs" / "nashhcc_s2_clusters"
VNN1_TOK = 4005
MIN_N = 20
TOP_K = 30


def log(m=""):
    print(m, flush=True)


def main():
    t0 = time.time()
    df = pd.read_csv(CLUSTERS_CSV)
    # cell_idx 是 tumor_mal 数据集内的索引；clusters CSV 按行序即数据集序
    cell_ids = df["cell_idx"].to_numpy()
    clusters = df["cluster"].astype(str).to_numpy()
    n = len(cell_ids)
    log(f"S2_hcc 细胞 {n}, 簇分布 {df['cluster'].value_counts().to_dict()}")

    from datasets import load_from_disk
    ds = load_from_disk(str(DS_DIR))
    all_ids = ds["input_ids"]
    total = len(all_ids)
    # 全数据集（3,693）的长度降序处理序——pickle 列表覆盖全部细胞，
    # 非 S2_hcc 细胞也必须占位（簇标签记 -1），否则列表位置会错位
    s2_of = {int(cid): i for i, cid in enumerate(cell_ids)}
    order = sorted(range(total), key=lambda i: -len(all_ids[i]))
    cluster_codes = {c: i for i, c in enumerate(sorted(set(clusters), key=int))}
    ncl = len(cluster_codes)

    # gene token -> 按处理序的数据集行号列表
    log("构建 gene->细胞 映射（全部 3,693 细胞）...")
    gene_rows = {}
    for i in order:
        for t in set(all_ids[i]):
            if t == VNN1_TOK or t < 0:
                continue
            gene_rows.setdefault(t, []).append(i)
    log(f"  涉及 gene token {len(gene_rows)} 个")

    d = pickle.load(open(PKL, "rb"))
    tok2name = {}
    gn = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    tokd = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    eid2name = {v: k for k, v in gn.items()}
    for eid, tv in tokd.items():
        if eid in eid2name:
            tok2name[tv] = eid2name[eid]

    log("聚合逐簇位移 ...")
    rows = []
    n_skip = 0
    for gtok, ilist in gene_rows.items():
        key = (VNN1_TOK, gtok)
        if key not in d:
            continue
        vals = np.asarray(d[key], dtype=float)
        if len(vals) != len(ilist):
            n_skip += 1
            continue
        # 每个位置对应的簇标签（非 S2_hcc 细胞 = -1）
        cl_arr = np.array([cluster_codes[clusters[s2_of[i]]] if i in s2_of else -1
                           for i in ilist])
        shift = 1.0 - vals
        keep = cl_arr >= 0
        cl_k = cl_arr[keep]
        sh_k = shift[keep]
        if len(sh_k) == 0:
            continue
        sums = np.bincount(cl_k, weights=sh_k, minlength=ncl)
        cnts = np.bincount(cl_k, minlength=ncl)
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        row = {"gene_token": gtok, "gene_name": tok2name.get(gtok, f"tok{gtok}")}
        for c, ci in cluster_codes.items():
            row[f"shift_c{c}"] = means[ci]
            row[f"n_c{c}"] = int(cnts[ci])
        row["global_shift"] = float(sh_k.mean())
        row["global_n"] = int(len(sh_k))
        rows.append(row)
    if n_skip:
        log(f"  ⚠ 跳过 {n_skip} 个长度不匹配的 token")
    g = pd.DataFrame(rows)
    g.to_csv(OUT_DIR / "cluster_gene_shifts_long.csv", index=False)
    log(f"长表: {len(g)} 基因 -> cluster_gene_shifts_long.csv")

    # 每簇 top 基因
    cl_keys = sorted(cluster_codes, key=int)
    top_rows = []
    for c in cl_keys:
        sub = g[g[f"n_c{c}"] >= MIN_N].dropna(subset=[f"shift_c{c}"])
        sub = sub.sort_values(f"shift_c{c}", ascending=False).head(TOP_K)
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            top_rows.append({
                "cluster": f"c{c}", "rank": rank, "gene": r["gene_name"],
                "shift_down": r[f"shift_c{c}"], "N_in_cluster": int(r[f"n_c{c}"]),
                "global_shift": r["global_shift"],
                "specificity": r[f"shift_c{c}"] - r["global_shift"],
            })
    top = pd.DataFrame(top_rows)
    top.to_csv(OUT_DIR / "cluster_top_genes.csv", index=False)

    # 簇特异性（差值最大）
    spec_rows = []
    for c in cl_keys:
        sub = g[(g[f"n_c{c}"] >= MIN_N)].dropna(subset=[f"shift_c{c}"]).copy()
        others = [f"shift_c{x}" for x in cl_keys if x != c]
        sub["rest_mean"] = sub[others].mean(axis=1)
        sub["diff"] = sub[f"shift_c{c}"] - sub["rest_mean"]
        for rank, (_, r) in enumerate(sub.sort_values("diff", ascending=False).head(15).iterrows(), 1):
            spec_rows.append({"cluster": f"c{c}", "rank": rank, "gene": r["gene_name"],
                              "shift_in_cluster": r[f"shift_c{c}"],
                              "shift_other_clusters_mean": r["rest_mean"],
                              "diff": r["diff"], "N_in_cluster": int(r[f"n_c{c}"])})
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(OUT_DIR / "cluster_specific_genes.csv", index=False)

    log("\n每簇位移最大 10 基因:")
    for c in cl_keys:
        s = top[top["cluster"] == f"c{c}"].head(10)
        log(f"  c{c}: " + ", ".join(f"{r.gene}({r.shift_down:.4f})" for r in s.itertuples()))
    log("\n每簇特异性最强 10 基因（相对其他簇）:")
    for c in cl_keys:
        s = spec[spec["cluster"] == f"c{c}"].head(10)
        log(f"  c{c}: " + ", ".join(f"{r.gene}(+{r.diff:.4f})" for r in s.itertuples()))

    log(f"\n总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
