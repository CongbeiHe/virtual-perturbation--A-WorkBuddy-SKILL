"""
步骤 3：用预训练 Geneformer 提取细胞 / 基因 embedding，并画 UMAP。

Geneformer 输出的 embedding 是「上下文化」的：同一个基因在不同细胞状态下的向量不同，
因此基因 embedding 可以直接用于发现共调控模块，细胞 embedding 可直接用于聚类注释。

【显存安全说明】
Geneformer V2-104M 的 config 声明 torch_dtype=float32，库内加载时不传 dtype，
在 8GB 显存笔记本上配合大 batch 会把显存榨干，进而触发显卡驱动崩溃（整机蓝屏重启）。
本脚本默认 bf16 + batch=16（约占 3.3GB），并在启动时做显存预检与自动降档。

用法：
    python scripts/02_extract_embeddings.py                                # 细胞 embedding（安全默认值）
    python scripts/02_extract_embeddings.py --mode gene                    # 基因 embedding
    python scripts/02_extract_embeddings.py --dtype fp32 --batch-size 16   # 需要更高数值精度时
    python scripts/02_extract_embeddings.py --quantized                    # 4bit 量化，最省显存
    python scripts/02_extract_embeddings.py --force-batch --batch-size 64  # 强制指定（风险自负）
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.geneformer_safe import (  # noqa: E402
    VRAMGuard,
    enable_bf16,
    recommend_batch_size,
    vram_report,
)

from geneformer import EmbExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Geneformer embedding 提取")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "V2-104M"))
    ap.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "tokenized" / "pbmc3k.dataset"))
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--prefix", default="pbmc3k_emb")
    ap.add_argument("--mode", default="cell", choices=["cell", "cls", "gene"])
    ap.add_argument("--max-ncells", type=int, default=None,
                    help="限制提取的细胞数；默认全部")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="前向批大小。8GB 显存建议 <=16(fp32) 或 <=32(bf16)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="推理精度。bf16 显存减半，为本机默认")
    ap.add_argument("--layer", type=int, default=-1, choices=[-1, 0],
                    help="-1 = 倒数第二层（更通用），0 = 最后一层")
    ap.add_argument("--label", default="cell_type", help="要附加到结果里的 .dataset 标签列")
    ap.add_argument("--quantized", action="store_true", help="使用 4bit 量化加载（最省显存）")
    ap.add_argument("--force-batch", action="store_true",
                    help="跳过显存预检，强制使用指定 batch size（不推荐）")
    args = ap.parse_args()

    # ---- 1) 精度护栏：让 Geneformer 以 bf16 加载，显存直接减半 ----
    if args.dtype in ("bf16", "fp16") and not args.quantized:
        enable_bf16(verbose=True)

    # ---- 2) 显存预检：把 batch size 压回物理显存的安全区间 ----
    dtype_factor = {"bf16": 1.0, "fp16": 1.0, "fp32": 2.0}[args.dtype]
    suggested = recommend_batch_size(dtype_factor=dtype_factor)

    import torch
    if torch.cuda.is_available():
        info = vram_report()
        print(f"设备: {info['device']}  总显存 {info['total_gb']}G  当前空闲 {info['free_gb']}G")
        print(f"精度: {args.dtype}   建议 batch <= {suggested}   指定 batch = {args.batch_size}")
        if args.batch_size > suggested and not args.force_batch:
            print(f"[vram-guard] 指定 batch {args.batch_size} 超出安全值，"
                  f"自动下调至 {suggested}（如需覆盖请加 --force-batch）")
            args.batch_size = suggested

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embex = EmbExtractor(
        model_type="Pretrained-Quantized" if args.quantized else "Pretrained",
        num_classes=0,
        emb_mode=args.mode,
        cell_emb_style="mean_pool",
        gene_emb_style="mean_pool",
        max_ncells=args.max_ncells,
        emb_layer=args.layer,
        emb_label=[args.label] if args.label else None,
        labels_to_plot=None,
        forward_batch_size=args.batch_size,
        nproc=8,
        model_version=args.model_version,
    )

    print(f"模型: {args.model}")
    print(f"数据: {args.input}")
    print(f"模式: {args.mode} embedding")

    # 注意：extract_embs 内部直接 load_from_disk(input_data_file)，
    # 因此这里必须传 .dataset 目录本身，而不是它的父目录。
    data_path = Path(args.input)
    if not data_path.exists():
        raise SystemExit(f"找不到 tokenized dataset: {data_path}")

    with VRAMGuard(limit_frac=0.90, tag="extract_embs") as guard:
        embs = embex.extract_embs(
            args.model,
            str(data_path),
            str(out_dir),
            args.prefix,
        )
        guard.check()

    print(f"\nembedding 维度: {embs.shape}")
    # Geneformer 的输出命名规则是 {prefix}.csv
    saved = out_dir / f"{args.prefix}.csv"
    print(f"已保存: {saved}")

    if args.mode == "gene":
        # 基因 embedding 支持用 scanpy 做 UMAP
        try:
            embex.plot_embs(
                embs=embs,
                plot_style="umap",
                output_directory=str(out_dir),
                output_prefix=args.prefix,
                gene_list=list(embs["Gene"][:2000]) if "Gene" in embs else None,
            )
            print(f"UMAP 图已保存到 {out_dir / 'figures'}")
        except Exception as exc:  # 绘图不是必需步骤
            print(f"[warn] 绘图跳过: {exc}")


if __name__ == "__main__":
    main()
