"""
步骤 4：计算机虚拟扰动（In Silico Perturbation）

原理
----
Geneformer 把细胞表示成「基因按表达量排序的 token 序列」。所谓虚拟扰动，就是
在**不碰湿实验**的前提下，直接在序列上做手术：

    "delete"      把目标基因从序列里删掉        —— 模拟基因敲除（knockout）
    "overexpress" 把目标基因挪到序列最前端      —— 模拟过表达

然后把扰动后的序列喂回模型，观察细胞 embedding（V2 用 CLS token）偏离原来多少。
偏离越大，说明这个基因对细胞状态的影响越关键。这是一次零样本的因果推断，
可以给出候选靶点排序，再拿去做湿实验验证。

两个阶段的产物
--------------
1. `perturb_data()`  → 中间 pickle 文件（每批细胞的原始位移数据）
2. `get_stats()`     → 最终 CSV（排序好的影响量化结果）

⚠️ **目录隔离**：`get_stats()` 会读取输入目录下的**全部** pickle 文件，
不会按前缀过滤。因此每次实验必须用独立目录，否则会把别的实验数据混进来
（表现为 `N_Detections` 异常偏高）。本脚本自动按 `--prefix` 建子目录：
`{outdir}/{prefix}/`。

⚠️ **精度陷阱**：默认 bf16 虽然省显存，但会把余弦相似度量化到约 1/512 的网格，
导致大量基因的 `Test_avg_shift` 退化成同一个数值（实测 1500 个基因只剩 6 个不同取值）。
做**结果排序/筛选**时建议加 `--dtype fp32`（实测峰值仅 1.41GB，代价约慢 50%）。

本脚本把两步串起来，并内置显存护栏（沿用 tools/geneformer_safe.py）。

⚠️ 重要约束
-----------
- **V2 模型必须用 `--emb-mode cls`**：因为 tokenize 时加了 `<cls>` 特殊 token，
  序列首位是 `<cls>`，此时用 "cell" 模式库内会直接报错。
- 计算量与 `基因数 × 细胞数` 成正比。全基因扰动非常耗时，
  因此默认用 `--max-ncells` 限制规模。

用法
----
    # 基础：逐个扰动每个基因，作用于 20 个细胞
    python scripts/04_in_silico_perturbation.py --genes all --max-ncells 20

    # 需要精确的基因影响排序时（推荐）
    python scripts/04_in_silico_perturbation.py --genes all --max-ncells 20 --dtype fp32

    # 过表达模式
    python scripts/04_in_silico_perturbation.py --perturb-type overexpress --genes all --max-ncells 20

    # 整组基因一起扰动（要求细胞共表达全部目标基因）
    python scripts/04_in_silico_perturbation.py --genes tcr --max-ncells 50

    # 自定义基因（ENSG ID 或基因符号，逗号分隔）
    python scripts/04_in_silico_perturbation.py --genes CD3D,CD3E,LCK --max-ncells 50

    # 按 dataset 列筛选细胞
    python scripts/04_in_silico_perturbation.py --genes all --max-ncells 30 --filter "cell_type:T_cell"
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.geneformer_safe import VRAMGuard, enable_bf16, vram_report  # noqa: E402

from geneformer import InSilicoPerturber, InSilicoPerturberStats  # noqa: E402

# ---------------------------------------------------------------- 预设基因集
# Ensembl ID -> 基因符号（供结果解读用）
ENSEMBL2SYMBOL = {
    # T 细胞
    "ENSG00000167286": "CD3D", "ENSG00000198851": "CD3E", "ENSG00000160654": "CD3G",
    "ENSG00000182866": "LCK", "ENSG00000115085": "ZAP70", "ENSG00000168685": "IL7R",
    "ENSG00000116824": "CD2", "ENSG00000178562": "CD28",
    # B 细胞
    "ENSG00000196092": "PAX5", "ENSG00000105369": "CD79A", "ENSG00000007312": "CD79B",
    "ENSG00000156738": "MS4A1", "ENSG00000177455": "CD19", "ENSG00000100721": "TCL1A",
    # 单核 / 巨噬
    "ENSG00000066336": "SPI1", "ENSG00000090382": "LYZ", "ENSG00000172216": "CEBPB",
    "ENSG00000170345": "FOS", "ENSG00000177606": "JUN", "ENSG00000143546": "S100A8",
    "ENSG00000163220": "S100A9", "ENSG00000203747": "FCGR3A",
    # NK
    "ENSG00000105374": "NKG7", "ENSG00000115523": "GNLY", "ENSG00000134539": "KLRD1",
    "ENSG00000180644": "PRF1",
    # 增殖
    "ENSG00000148773": "MKI67", "ENSG00000131747": "TOP2A", "ENSG00000132646": "PCNA",
    # 管家基因（阴性对照：扰动它们的影响应偏小或无方向性）
    "ENSG00000075624": "ACTB", "ENSG00000111640": "GAPDH", "ENSG00000166710": "B2M",
    "ENSG00000142541": "RPL13A",
}

# 共表达模块：同一谱系内常共表达的基因，可用于「整模块一起扰动」的列表模式。
# 注意：列表模式要求细胞同时包含全部目标基因（库内用 == len(tokens) 判定），
#      因此不能把互斥的谱系 marker 混在一起。
COEXPRESS_MODULES = {
    "tcr":  ["ENSG00000167286", "ENSG00000198851", "ENSG00000160654",
             "ENSG00000182866", "ENSG00000115085"],           # CD3D CD3E CD3G LCK ZAP70
    "bcr":  ["ENSG00000196092", "ENSG00000105369", "ENSG00000007312"],  # PAX5 CD79A CD79B
    "myeloid": ["ENSG00000090382", "ENSG00000143546", "ENSG00000163220"],  # LYZ S100A8 S100A9
    "nk":   ["ENSG00000105374", "ENSG00000115523", "ENSG00000134539"],  # NKG7 GNLY KLRD1
    "housekeeping": ["ENSG00000075624", "ENSG00000111640"],   # ACTB GAPDH
}


def parse_genes(spec: str) -> "str | list":
    """把 --genes 参数解析成 InSilicoPerturber 需要的形式。

    - "all"    : 逐个扰动每个细胞中检测到的每个基因（标准用法，计算量大）
    - 模块名    : 见 COEXPRESS_MODULES，整组基因一起扰动
    - 逗号列表  : ENSG ID 或基因符号，整组一起扰动
    """
    s = spec.strip()
    if s.lower() == "all":
        return "all"
    if s.lower() in COEXPRESS_MODULES:
        return list(COEXPRESS_MODULES[s.lower()])
    # 支持逗号分隔；允许用基因符号，自动转 Ensembl ID
    items = [x.strip() for x in s.split(",") if x.strip()]
    resolved, unknown = [], []
    for x in items:
        if x.upper().startswith("ENSG"):
            resolved.append(x.upper())
        else:
            hit = next((k for k, v in ENSEMBL2SYMBOL.items() if v.upper() == x.upper()), None)
            if hit:
                resolved.append(hit)
            else:
                unknown.append(x)
    if unknown:
        print(f"[warn] 以下基因无法识别，已忽略: {', '.join(unknown)}")
    return resolved


def main():
    ap = argparse.ArgumentParser(description="Geneformer 计算机虚拟扰动")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "V2-104M"))
    ap.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "tokenized" / "pbmc3k.dataset"),
                    help="tokenized .dataset 目录")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "outputs" / "isp"))
    ap.add_argument("--stats-outdir", default=str(PROJECT_ROOT / "outputs" / "isp_stats"))
    ap.add_argument("--prefix", default="pbmc3k")
    ap.add_argument("--perturb-type", default="delete",
                    choices=["delete", "overexpress", "inhibit", "activate"])
    ap.add_argument("--genes", default="all",
                    help='"all"(逐个扰动每个基因，标准用法) / 模块名(tcr|bcr|myeloid|nk|housekeeping) / ENSG或基因符号逗号分隔')
    ap.add_argument("--max-ncells", type=int, default=20,
                    help="参与扰动的细胞数上限。计算量 ≈ 细胞数 x 每细胞基因数")
    ap.add_argument("--emb-mode", default="cls", choices=["cls", "cell", "cls_and_gene", "cell_and_gene"],
                    help="V2 模型必须用 cls（序列首位是 <cls> token）")
    ap.add_argument("--layer", type=int, default=0, choices=[-1, 0],
                    help="0 = 最后一层（贴合训练目标），-1 = 倒数第二层（更通用）")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="前向批大小，8GB 显存建议 <=16")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--filter", default=None,
                    help='按 .dataset 列筛选细胞，格式 "列名:值1,值2"，如 "leiden:5,7"')
    ap.add_argument("--stats-mode", default="auto",
                    choices=["auto", "aggregate_data", "mixture_model", "vs_null",
                             "goal_state_shift", "aggregate_gene_shifts"],
                    help="结果统计模式。auto = 按 genes 类型自动选："
                         "all -> mixture_model；基因列表 -> aggregate_data")
    ap.add_argument("--skip-stats", action="store_true", help="只生成中间 pickle，不做统计")
    ap.add_argument("--stats-only", action="store_true",
                    help="跳过阶段 1，直接用目录里已有的中间 pickle 做统计"
                         "（阶段 1 中断或卡死时的补救手段）")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                    help="推理精度。默认 fp32 —— 余弦相似度可分辨到 1e-8，"
                         "基因影响能真正排出先后；bf16 快 50%% 但位移值会被量化到 "
                         "约 1/512 网格（实测 1500 基因只剩 6 个不同取值），"
                         "仅适合快速验证流程")
    args = ap.parse_args()

    # ---- 显存护栏 ----
    if args.dtype == "bf16":
        enable_bf16()
    if vram_report()["available"]:
        info = vram_report()
        print(f"设备: {info['device']}  总显存 {info['total_gb']}G")
        if args.batch_size > 16:
            print(f"[vram-guard] batch {args.batch_size} 偏大，已下调至 16")
            args.batch_size = 16

    dataset_dir = Path(args.input)
    if not dataset_dir.exists():
        raise SystemExit(f"找不到 tokenized dataset: {dataset_dir}")

    genes = parse_genes(args.genes)
    n_genes = len(genes) if isinstance(genes, list) else "全部"

    # filter 参数解析
    filter_data = None
    if args.filter:
        col, vals = args.filter.split(":", 1)
        filter_data = {col.strip(): [v.strip() for v in vals.split(",")]}

    out_dir = Path(args.outdir) / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_outdir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Geneformer 计算机虚拟扰动")
    print("=" * 68)
    print(f"模型        : {args.model}")
    print(f"数据        : {dataset_dir}")
    print(f"扰动类型    : {args.perturb_type}")
    print(f"目标基因    : {n_genes} 个")
    print(f"细胞上限    : {args.max_ncells}")
    print(f"emb 模式    : {args.emb_mode}   层: {args.layer}")
    print(f"batch size  : {args.batch_size}")
    if filter_data:
        print(f"细胞筛选    : {filter_data}")
    print("=" * 68)

    # ---------------- 细胞可用性预检 ----------------
    # 列表模式下库内要求细胞「同时包含全部目标基因」(== len(tokens))，
    # 而谱系 marker 互斥，因此先预检有多少细胞可用，避免白跑。
    try:
        import pickle

        from datasets import load_from_disk

        ds = load_from_disk(str(dataset_dir))
        n_cells_total = len(ds)
        print(f"数据集共 {n_cells_total} 个细胞")

        if isinstance(genes, list):
            td_name = "token_dictionary_gc104M.pkl" if args.model_version == "V2" else "token_dictionary_gc30M.pkl"
            td_file = PROJECT_ROOT / "code" / "geneformer" / td_name
            if td_file.exists():
                td = pickle.load(open(td_file, "rb"))
                gene_tokens = {td[g] for g in genes if g in td}
                missing = [g for g in genes if g not in td]
                if missing:
                    print(f"[warn] {len(missing)} 个基因无 token，将被忽略")
                if not gene_tokens:
                    raise SystemExit("目标基因全部无 token，无法扰动")

                n_ok = 0
                step = 500
                for s in range(0, n_cells_total, step):
                    for ids in ds["input_ids"][s : s + step]:
                        if gene_tokens.issubset(set(ids)):
                            n_ok += 1
                print(f"同时包含全部 {len(gene_tokens)} 个目标基因的细胞: {n_ok} 个")
                if n_ok == 0:
                    raise SystemExit(
                        "没有任何细胞同时表达这组基因，无法扰动。\n"
                        "  原因：列表模式要求细胞包含全部目标基因（库内用 == len(tokens) 判定），\n"
                        "  而互斥的谱系 marker 不会共存于同一细胞。\n"
                        "  解决：改用 --genes all 逐个扰动；或只选同谱系共表达的基因。"
                    )
                if n_ok < args.max_ncells:
                    print(f"[vram-guard] 可用细胞仅 {n_ok} 个，自动下调 max_ncells")
                    args.max_ncells = n_ok
        else:
            sample = min(200, n_cells_total)
            lens = [len(x) for x in ds["input_ids"][:sample]]
            avg_len = sum(lens) / len(lens)
            est = args.max_ncells * avg_len
            print(f"平均序列长度 {avg_len:.0f} token（抽样 {sample} 个细胞）")
            print(f"预计前向次数 ≈ {args.max_ncells} 细胞 x {avg_len:.0f} 基因 = {est:,.0f} 次")
            if est > 50000:
                print("[提示] 计算量偏大，可用 --max-ncells 调小，或耐心等待")
    except SystemExit:
        raise
    except Exception as exc:  # 预检失败不应阻断主流程
        print(f"[warn] 预检跳过: {type(exc).__name__}: {exc}")

    print("=" * 68)

    # ---------------- 阶段 1：计算扰动位移 ----------------
    isp = InSilicoPerturber(
        perturb_type=args.perturb_type,
        perturb_rank_shift=None,
        genes_to_perturb=genes,
        combos=0,
        anchor_gene=None,
        model_type="Pretrained",
        num_classes=0,
        emb_mode=args.emb_mode,
        cell_emb_style="mean_pool",
        filter_data=filter_data,
        cell_states_to_model=None,
        state_embs_dict=None,
        max_ncells=args.max_ncells,
        cell_inds_to_perturb="all",
        emb_layer=args.layer,
        forward_batch_size=args.batch_size,
        nproc=args.nproc,
        model_version=args.model_version,
    )

    if args.stats_only:
        pkls = sorted(out_dir.glob(f"in_silico_{args.perturb_type}_{args.prefix}*"))
        if not pkls:
            raise SystemExit(
                f"--stats-only 模式：在 {out_dir} 下没找到中间 pickle。\n"
                f"  请先正常跑一次（不带 --stats-only）生成中间文件。"
            )
        print(f"\n[跳过阶段 1] 直接复用已有的 {len(pkls)} 个中间文件")
        print(f"目录: {out_dir}")
    else:
        print("\n[阶段 1/2] 计算扰动位移（生成中间 pickle）...")
        t0 = time.time()
        with VRAMGuard(limit_frac=0.90, tag="perturb_data") as guard:
            isp.perturb_data(args.model, str(dataset_dir), str(out_dir), args.prefix)
            guard.check()
        print(f"[阶段 1/2] 完成，用时 {time.time()-t0:.1f}s")

        # 列出中间产物
        pkls = sorted(out_dir.glob(f"in_silico_{args.perturb_type}_{args.prefix}*"))
        print(f"中间文件 {len(pkls)} 个，位于 {out_dir}")

    if args.skip_stats:
        print("\n已跳过统计阶段（--skip-stats）")
        return

    # ---------------- 阶段 2：统计汇总 ----------------
    # 模式兼容性（库内硬性校验）：
    #   genes_perturbed == "all"  -> 只能用 mixture_model / aggregate_gene_shifts
    #   genes_perturbed == list   -> 只能用 aggregate_data 等
    stats_mode = args.stats_mode
    if stats_mode == "auto":
        stats_mode = "mixture_model" if genes == "all" else "aggregate_data"

    if stats_mode == "mixture_model" and genes != "all":
        raise SystemExit(
            "mixture_model 要求 genes_perturbed='all'。\n"
            "  请改用 --stats-mode aggregate_data，或把 --genes 设为 all。"
        )
    if stats_mode == "aggregate_data" and genes == "all":
        raise SystemExit(
            "aggregate_data 要求 genes_perturbed 是具体基因列表（不能是 all）。\n"
            "  请改用 --stats-mode mixture_model，或指定具体基因。"
        )

    print(f"\n[阶段 2/2] 汇总统计（mode={stats_mode}）...")
    ispstats = InSilicoPerturberStats(
        mode=stats_mode,
        genes_perturbed=genes,
        combos=0,
        anchor_gene=None,
        cell_states_to_model=None,
        pickle_suffix="_raw.pickle",
        model_version=args.model_version,
    )

    t0 = time.time()
    ispstats.get_stats(str(out_dir), None, str(stats_dir), args.prefix)
    print(f"[阶段 2/2] 完成，用时 {time.time()-t0:.1f}s")

    csvs = sorted(stats_dir.glob(f"*{args.prefix}*.csv"))
    print(f"\n结果 CSV {len(csvs)} 个:")
    for c in csvs:
        print(f"  {c}")

    # 附加基因符号映射，方便解读
    mapping_file = stats_dir / f"{args.prefix}_ensembl2symbol.json"
    mapping_file.write_text(json.dumps(ENSEMBL2SYMBOL, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n基因符号映射已写出: {mapping_file}")
    print("\n下一步: python scripts/05_analyze_perturbation.py")


if __name__ == "__main__":
    main()
