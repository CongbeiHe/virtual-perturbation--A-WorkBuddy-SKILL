"""rank-matched null 系列脚本的公共工具（供 40-47 号脚本复用）。

路径解析约定（优先级从高到低）
------------------------------
1. 显式 `--workdir` / `--code-dir` 参数
2. 环境变量 `GENEFORMER_PERT_WORKDIR` / `GENEFORMER_CODE_DIR`
3. 当前工作目录（workdir）；code-dir 从已安装的 geneformer 推导，
   推导失败再回退到 `<repo>/code/geneformer`

`geneformer_pert` 的 `OUTPUT_DIR` 在 import 时就从 `GENEFORMER_PERT_WORKDIR`
读定，因此**必须在 import geneformer_pert 之前**设置该环境变量
（见 45 号脚本顶部）。
"""
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def workdir(explicit=None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("GENEFORMER_PERT_WORKDIR")
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def code_dir(explicit=None) -> Path:
    """官方 geneformer 源码目录（含 token_dictionary_gc104M.pkl）。"""
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("GENEFORMER_CODE_DIR")
    if env:
        return Path(env).resolve()
    try:
        import geneformer  # noqa: F401
        return Path(geneformer.__file__).resolve().parent
    except Exception:
        return Path(__file__).resolve().parent.parent / "code" / "geneformer"


def out_dirs(wd: Path):
    """(data/, outputs/)，不存在则创建。"""
    d, o = Path(wd) / "data", Path(wd) / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    o.mkdir(parents=True, exist_ok=True)
    return d, o


def load_dicts(cdir: Path):
    """返回 (token_dictionary, gene_name_id_dict)。

    ⚠ 两个字典都要查。事故（2026-11-01）：LINC02256 (ENSG00000261064)
    在 gene_name_id_dict 里有 ENSG，但 token_dictionary 里查无此 token，
    导致 `tok[target_eid]` 抛 KeyError、整批 run 白跑。只校验前者会漏。
    """
    tok = pickle.load(open(Path(cdir) / "token_dictionary_gc104M.pkl", "rb"))
    g2i = pickle.load(open(Path(cdir) / "gene_name_id_dict_gc104M.pkl", "rb"))
    return tok, g2i


def col(X, j):
    """取矩阵第 j 列为 1-D ndarray（兼容稀疏）。"""
    c = X[:, j]
    return np.asarray(c.todense()).ravel() if sp.issparse(c) else np.asarray(c).ravel()


def counts_matrix(adata):
    """优先 layers['counts']，否则 X；返回 CSR。"""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return X.tocsr() if sp.issparse(X) else X


def gene_symbols(adata):
    """基因符号数组：优先 var['feature_name']，否则 var_names。"""
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).to_numpy()
    return np.asarray(adata.var_names).astype(str)


def find_gene(sym, name):
    """按 HGNC 符号（大小写不敏感）找列号，找不到返回 -1。"""
    hit = np.where(np.char.upper(sym.astype(str)) == name.upper())[0]
    return int(hit[0]) if len(hit) else -1


def row_rank_stats(sub, n_genes):
    """逐行降序 ordinal rank（1-based），按基因汇总 (检出数, rank 中位)。

    与 Geneformer 的分词语序一致：仅对非零（表达）的基因给 rank。
    """
    indptr, indices, data = sub.indptr, sub.indices, sub.data
    ranks = np.empty(len(data), dtype=np.int32)
    for r in range(sub.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        d = data[s:e]
        k = e - s
        order = np.argsort(-d, kind="stable")
        rk = np.empty(k, dtype=np.int32)
        rk[order] = np.arange(1, k + 1, dtype=np.int32)
        ranks[s:e] = rk
    o = np.argsort(indices, kind="stable")
    g, r = indices[o], ranks[o]
    uniq, starts, cnts = np.unique(g, return_index=True, return_counts=True)
    det = np.zeros(n_genes, dtype=np.int32)
    med = np.full(n_genes, np.nan)
    det[uniq] = cnts.astype(np.int32)
    med[uniq] = [np.median(r[s:s + c]) for s, c in zip(starts, cnts)]
    return det, med


def target_rank_in_cells(sub, expr, idx):
    """目标基因在 target⁺ 细胞中各自的 rank（1-based）。

    rank = 1 + 该细胞内表达量严格大于目标基因的基因数。
    """
    tr = np.empty(len(idx), dtype=np.int32)
    for k, i in enumerate(idx):
        tr[k] = 1 + int((sub[k].data > expr[i]).sum())
    return tr
