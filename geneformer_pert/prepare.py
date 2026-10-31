"""步骤一：从原始 h5ad 抽取细胞子集并转为 Geneformer 可用格式。

移植自 scripts/21_prepare_nashhcc_tumor.py，逻辑一致，参数化：
  - cell_filter: pandas query 字符串（作用于 obs），如 "malignant and Tissue=='Tumor'"
  - 自动映射基因符号 -> gc104M Ensembl，去重
  - 输出 h5ad（counts int32, var index=Ensembl, feature_name 列）
"""
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import GENE_NAME_ID_DICT, MIN_NCOUNTS


def log(m=""):
    print(m, flush=True)


def _pick_counts(a, counts_layer=None):
    """从 AnnData 中取整数 counts 矩阵。

    优先级：显式 counts_layer > layers['counts'] > X（须为整数型）。
    兼容不同数据集的结构差异。
    """
    if counts_layer is not None:
        assert counts_layer in a.layers, \
            f"指定 counts_layer='{counts_layer}' 不存在；可用 layers: {list(a.layers)}"
        counts = a.layers[counts_layer]
    elif "counts" in a.layers:
        counts = a.layers["counts"]
    else:
        counts = a.X
        data = counts.data if sp.issparse(counts) else np.asarray(counts).ravel()
        assert np.allclose(data, np.round(data)), \
            (f"X 非整数 counts 且无 layers['counts']。可用 layers: {list(a.layers)}；"
             "请用 --counts-layer 显式指定原始计数层")
    return counts


def _pick_symbols(b, gene_col=None):
    """取基因符号数组。自动识别 var_names 是符号还是 Ensembl ID。"""
    if gene_col is not None:
        assert gene_col in b.var.columns, \
            f"指定 gene_col='{gene_col}' 不在 var 列中: {list(b.var.columns)}"
        return b.var[gene_col].astype(str).to_numpy()
    names = b.var_names.astype(str).to_numpy()
    n_ens = sum(1 for s in names[: min(200, len(names))] if str(s).startswith("ENSG"))
    if n_ens > len(names[:200]) / 2:                       # var_names 是 Ensembl
        for c in ["gene_name", "gene_symbols", "symbol", "feature_name", "_index"]:
            if c in b.var.columns:
                log(f"  var_names 为 Ensembl，基因符号取自 var['{c}']")
                return b.var[c].astype(str).to_numpy()
        log("  ⚠ var_names 为 Ensembl 且无符号列，直接按 Ensembl 映射")
        return names
    return names


def prepare_subset(src_h5ad, cell_filter, dst_dir, dataset_name,
                   keep_attrs=None, target_gene=None, force=False,
                   counts_layer=None, gene_col=None):
    """抽取细胞子集 -> {dst_dir}/{dataset_name}.h5ad。

    cell_filter 示例（作用于 obs，列名因数据集而异，先用探针脚本确认）:
        "malignant and Tissue=='Tumor'"           肿瘤恶性细胞
        "cell_type == 'Hepatocyte'"               按细胞类型
    counts_layer / gene_col: 适配非标准 h5ad（默认自动探测）。
    返回输出 h5ad 路径。
    """
    src_h5ad = Path(src_h5ad)
    dst_dir = Path(dst_dir)
    out_path = dst_dir / f"{dataset_name}.h5ad"
    if out_path.exists() and not force:
        log(f"[skip] {out_path} 已存在")
        return out_path
    dst_dir.mkdir(parents=True, exist_ok=True)
    if keep_attrs is None:
        keep_attrs = ["cell_type", "SampleID", "Patient", "Tissue", "Condition",
                      "patient", "tissue", "group", "malignant"]

    log(f"读取 {src_h5ad}")
    a = ad.read_h5ad(src_h5ad)
    log(f"  原始: {a.shape[0]} 细胞 × {a.shape[1]} 基因")

    mask = a.obs.eval(cell_filter).to_numpy()
    log(f"  cell_filter='{cell_filter}' -> {mask.sum()} / {len(mask)} 细胞")
    assert mask.sum() > 0, "过滤条件没有命中任何细胞"
    b = a[mask].copy()

    counts = _pick_counts(b, counts_layer)
    if sp.issparse(counts):
        counts = counts.tocsr()
    data = counts.data if sp.issparse(counts) else np.asarray(counts).ravel()
    assert np.allclose(data, np.round(data)), "counts 不是整数！"
    counts = counts.astype(np.int32)
    n_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    keep_cells = n_counts >= MIN_NCOUNTS
    if not keep_cells.all():
        log(f"  剔除 {(~keep_cells).sum()} 个零计数细胞")
        counts, b = counts[keep_cells], b[keep_cells]
        n_counts = n_counts[keep_cells]

    g2i = pickle.load(open(GENE_NAME_ID_DICT, "rb"))
    symbols = _pick_symbols(b, gene_col)
    ensembl = np.array([g2i.get(s, s if str(s).startswith("ENSG") else "") for s in symbols],
                       dtype=object)
    keep = ensembl != ""
    log(f"  基因符号可映射到 gc104M: {keep.sum()} / {len(symbols)}")
    counts = counts[:, keep]
    new_var = pd.DataFrame(index=pd.Index(ensembl[keep].astype(str), name=None))
    new_var["feature_name"] = symbols[keep]
    dup = new_var.index.duplicated()
    if dup.any():
        log(f"  去重 {int(dup.sum())} 个重复 Ensembl")
        counts, new_var = counts[:, ~dup], new_var[~dup]

    obs = pd.DataFrame(index=pd.Index(b.obs_names.astype(str)))
    obs["n_counts"] = n_counts
    for c in keep_attrs:
        if c in b.obs.columns:
            obs[c] = b.obs[c].astype(str).values

    out = ad.AnnData(X=counts, obs=obs, var=new_var)
    if "X_umap" in b.obsm:
        out.obsm["X_umap"] = np.asarray(b.obsm["X_umap"])

    if target_gene is not None:
        if target_gene in list(new_var.index):
            j = list(new_var.index).index(target_gene)
        elif target_gene in set(new_var["feature_name"]):
            j = list(new_var["feature_name"]).index(target_gene)
        else:
            j = None
        if j is not None:
            vcol = (np.asarray(counts[:, j].todense()).ravel()
                    if sp.issparse(counts) else np.asarray(counts[:, j]).ravel())
            log(f"  {target_gene}: 表达细胞 {int((vcol>0).sum())}/{len(vcol)} "
                f"({(vcol>0).mean():.1%})  max={vcol.max():.0f}")
        else:
            log(f"  ⚠ 目标基因 {target_gene} 不在数据中（检查符号/Ensembl）")

    out.write_h5ad(out_path, compression="gzip")
    log(f"  已保存 -> {out_path}  ({out.shape[0]} 细胞 × {out.shape[1]} 基因)")
    return out_path
