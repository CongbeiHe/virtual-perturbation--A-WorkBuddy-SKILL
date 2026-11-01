"""
步骤 1：把 10x Genomics 原始表达矩阵转换为 Geneformer 要求的 .h5ad 格式。

Geneformer tokenizer 对输入的两条硬性要求：
  1. 基因维（var）必须能映射到 Ensembl ID —— 10x 输出的是基因符号（gene symbol），
     Geneformer 自带的映射字典可以直接把符号转成 Ensembl ID（含大量别名），
     因此这里只需保证 var_names 是基因符号，并用 use_h5ad_index=True 交给 tokenizer 处理。
  2. 细胞维（obs）必须有 n_counts 列（每个细胞的总 UMI 数），用于表达值归一化。

用法：
    python scripts/00_prepare_pbmc3k.py
"""

from pathlib import Path

import numpy as np
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "filtered_gene_bc_matrices" / "hg19"
OUT_FILE = PROJECT_ROOT / "data" / "pbmc3k_raw.h5ad"


def main():
    if not RAW_DIR.exists():
        raise SystemExit(
            f"未找到 10x 原始数据目录: {RAW_DIR}\n"
            "请先执行: curl -L -o pbmc3k.tar.gz "
            "https://cf.10xgenomics.com/samples/cell/pbmc3k/"
            "pbmc3k_filtered_gene_bc_matrices.tar.gz && tar -xzf pbmc3k.tar.gz -C data/"
        )

    print(f"读取 10x 矩阵: {RAW_DIR}")
    adata = sc.read_10x_mtx(RAW_DIR, var_names="gene_symbols", cache=False)
    print(f"原始维度: {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    # 必须：每个细胞的总计数（用于 size factor 归一化）
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel().astype(int)

    # 可选：自定义细胞属性，tokenize 时可通过 custom_attr_name_dict 带进 .dataset
    adata.obs["cell_type"] = "PBMC"

    # 保留原始计数（Geneformer 需要 raw counts，不要做 log1p / scale）
    adata.var["gene_symbol"] = adata.var_names.astype(str)

    adata.write_h5ad(OUT_FILE, compression="gzip")
    print(f"已写出: {OUT_FILE}")
    print(f"总 UMI 中位数: {int(np.median(adata.obs['n_counts']))}")


if __name__ == "__main__":
    main()
