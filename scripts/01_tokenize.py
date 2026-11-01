"""
步骤 2：把 .h5ad 原始计数转换成 Geneformer 的 rank value encoding（.dataset）。

核心思想：对每个细胞，把基因按「该基因在本细胞的表达 / 该基因在 1 亿细胞预训练语料中的中位表达」
降序排列，只保留非零基因，得到一串基因 token 序列。这样高频管家基因会被压到低排名，
而真正区分细胞状态的转录因子会被顶到前面。

用法：
    python scripts/01_tokenize.py                       # 默认 V2（4096 输入长度）
    python scripts/01_tokenize.py --model-version V1    # V1 系列（2048 输入长度）
    python scripts/01_tokenize.py --input 某个.h5ad --prefix mydata
"""

import argparse
from pathlib import Path

from geneformer import TranscriptomeTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Geneformer 转录组tokenizer")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "pbmc3k_raw.h5ad"),
                    help="输入 .h5ad / .loom 文件或目录")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "data" / "tokenized"))
    ap.add_argument("--prefix", default="pbmc3k")
    ap.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--attrs", default="cell_type",
                    help="要带进 .dataset 的 obs 列，逗号分隔；留空表示不带")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输入可以是单文件（复制到一个临时目录）或目录
    if input_path.is_file():
        work_dir = out_dir / "_input"
        work_dir.mkdir(exist_ok=True)
        target = work_dir / input_path.name
        if not target.exists():
            import shutil
            shutil.copy2(input_path, target)
        data_dir = work_dir
    else:
        data_dir = input_path

    custom_attr = {a: a for a in (args.attrs.split(",") if args.attrs else []) if a}

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr or None,
        nproc=args.nproc,
        model_version=args.model_version,
        # 基因索引是 gene symbol，交给 Geneformer 自带映射字典转 Ensembl ID
        use_h5ad_index=True,
        special_token=True,
    )
    tk.tokenize_data(
        str(data_dir),
        str(out_dir),
        args.prefix,
        file_format="h5ad",
    )

    out_path = out_dir / f"{args.prefix}.dataset"
    print(f"\n完成 -> {out_path}")


if __name__ == "__main__":
    main()
