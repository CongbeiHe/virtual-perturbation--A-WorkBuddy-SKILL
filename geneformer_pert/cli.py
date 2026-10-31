"""命令行入口。

用法:
  python -m geneformer_pert prepare --src task/GSE189175/GSE189175_NASHHCC.h5ad \
      --cell-filter "malignant and Tissue=='Tumor'" --dataset-name myset \
      --target VNN1
  python -m geneformer_pert perturb --dataset-name myset --target VNN1 --mode overexpress
  python -m geneformer_pert displacement --dataset-name myset --target VNN1 --mode overexpress
  python -m geneformer_pert all ...  (prepare -> perturb -> displacement)
"""
import argparse

from .config import PERTURB_MODES


def add_common(p):
    p.add_argument("--dataset-name", required=True, help="数据集名（决定输出目录）")
    p.add_argument("--target", required=True, help="目标基因符号或 Ensembl ID")
    p.add_argument("--mode", default="overexpress", choices=list(PERTURB_MODES))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workdir", default=None,
                   help="输出根目录（默认 = GENEFORMER_HOME；跨工作空间时指向当前工作空间）")


def apply_workdir(path):
    if path:
        import os
        os.environ["GENEFORMER_PERT_WORKDIR"] = str(path)
        import importlib
        from . import config
        importlib.reload(config)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geneformer_pert",
                                 description="Geneformer 虚拟扰动统一流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare", help="抽取细胞子集并转为 Geneformer 格式")
    add_common(p1)
    p1.add_argument("--src", required=True, help="原始 h5ad 路径")
    p1.add_argument("--cell-filter", required=True,
                    help="pandas query，作用于 obs，如 \"malignant and Tissue=='Tumor'\"")
    p1.add_argument("--counts-layer", default=None,
                    help="counts 层名（默认自动探测 layers['counts'] 或整数 X）")
    p1.add_argument("--gene-col", default=None,
                    help="var 中基因符号列名（默认自动识别 var_names 类型）")
    p1.add_argument("--force", action="store_true")

    p2 = sub.add_parser("perturb", help="tokenize + ISP + 统计表")
    add_common(p2)

    p3 = sub.add_parser("displacement", help="UMAP 位移 + 凝聚度 + 随机基线")
    add_common(p3)
    p3.add_argument("--no-plots", action="store_true")

    p4 = sub.add_parser("all", help="prepare -> perturb -> displacement")
    add_common(p4)
    p4.add_argument("--src", required=True)
    p4.add_argument("--cell-filter", required=True)
    p4.add_argument("--counts-layer", default=None)
    p4.add_argument("--gene-col", default=None)
    p4.add_argument("--no-plots", action="store_true")

    a = ap.parse_args(argv)
    apply_workdir(a.workdir)

    if a.cmd == "prepare":
        from .prepare import prepare_subset
        from .config import default_dirs
        dirs = default_dirs(a.dataset_name, a.target, a.mode)
        prepare_subset(a.src, a.cell_filter, dirs["counts_dir"], a.dataset_name,
                       target_gene=a.target, force=a.force,
                       counts_layer=a.counts_layer, gene_col=a.gene_col)

    elif a.cmd == "perturb":
        from .pipeline import run_isp
        run_isp(a.dataset_name, a.target, a.mode, batch=a.batch)

    elif a.cmd == "displacement":
        from .displacement import run_displacement
        run_displacement(a.dataset_name, a.target, a.mode, batch=a.batch,
                         make_plots=not a.no_plots)

    elif a.cmd == "all":
        from .config import default_dirs
        from .prepare import prepare_subset
        from .pipeline import run_isp
        from .displacement import run_displacement
        dirs = default_dirs(a.dataset_name, a.target, a.mode)
        prepare_subset(a.src, a.cell_filter, dirs["counts_dir"], a.dataset_name,
                       target_gene=a.target, counts_layer=a.counts_layer,
                       gene_col=a.gene_col)
        run_isp(a.dataset_name, a.target, a.mode, batch=a.batch)
        run_displacement(a.dataset_name, a.target, a.mode, batch=a.batch,
                         make_plots=not a.no_plots)


if __name__ == "__main__":
    main()
