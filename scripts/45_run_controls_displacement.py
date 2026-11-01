"""批量为 rank-matched 对照基因只跑 displacement 阶段（省一半 GPU 时间）。

为什么可以跳过 ISP
------------------
完整 L1 链 = prepare + ISP + displacement，后两者各占约一半。
`displacement.py` 只在**交叉验证**时读 ISP 的 raw pickle（glob 查找），
缺失则 xval=None 直接跳过，不影响 `per_cell_shift.csv` 的产出。

代价：这批对照没有官方 ISP 对拍值。建议至少保留 3-4 个走完整链（含 ISP）
作锚点，其余走本脚本。

⚠ 必须单进程串行：8GB 显存下多进程必 OOM。

用法
----
  # 先设置工作目录（决定输出落在哪）
  export GENEFORMER_PERT_WORKDIR=/path/to/workdir      # Linux
  $env:GENEFORMER_PERT_WORKDIR="..."                   # PowerShell
  python scripts/45_run_controls_displacement.py \
      --datasets Atrial15k_hif3ap_s2k --genes-file outputs/hif3a20_genes.txt

中断后重跑会自动跳过已完成项。
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

# ⚠ geneformer_pert 的 OUTPUT_DIR 在 import 时就从该环境变量读定，
#    因此必须在 import 之前设置，否则输出会落到默认位置。
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--workdir", default=None)
_a, _ = _ap.parse_known_args()
if _a.workdir:
    os.environ["GENEFORMER_PERT_WORKDIR"] = str(Path(_a.workdir).resolve())
elif not os.environ.get("GENEFORMER_PERT_WORKDIR"):
    os.environ["GENEFORMER_PERT_WORKDIR"] = str(Path.cwd().resolve())

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import geneformer_pert as gp  # noqa: E402
from geneformer_pert.config import default_dirs  # noqa: E402
from _ranknull import workdir  # noqa: E402


def already_done(analysis_dir: Path) -> bool:
    p = Path(analysis_dir) / "per_cell_shift.csv"
    if not p.exists():
        return False
    try:
        return len(pd.read_csv(p)) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="对照 run 所用数据集名（target+ 子集或其子样本）")
    ap.add_argument("--genes", nargs="*", default=None)
    ap.add_argument("--genes-file", default=None,
                    help="每行一个基因名的文本文件")
    ap.add_argument("--mode", default="overexpress")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    genes = list(a.genes or [])
    if a.genes_file:
        genes += [g.strip() for g in Path(a.genes_file).read_text(
            encoding="utf-8").splitlines() if g.strip()]
    if not genes:
        raise SystemExit("无对照基因；用 --genes 或 --genes-file，"
                         "或先跑 44/47 号脚本生成清单")

    jobs = [(ds, g) for ds in a.datasets for g in genes]
    print(f"待跑 {len(jobs)} 项：{len(a.datasets)} 数据集 × {len(genes)} 基因",
          flush=True)
    print(f"基因 = {', '.join(genes)}", flush=True)
    print(f"workdir = {workdir(a.workdir)}", flush=True)

    if a.dry_run:
        for ds, g in jobs:
            d = Path(default_dirs(ds, g, a.mode)["analysis"])
            print(f"  [{'skip' if already_done(d) else 'run '}] {d}")
        return

    t_start = time.time()
    for k, (ds, g) in enumerate(jobs, 1):
        dirs = default_dirs(ds, g, a.mode)
        if already_done(Path(dirs["analysis"])):
            print(f"[{k}/{len(jobs)}] skip {ds} {g}（已完成）", flush=True)
            continue
        print(f"\n=== [{k}/{len(jobs)}] {ds} {g}  "
              f"({time.strftime('%F %T')}) ===", flush=True)
        t0 = time.time()
        try:
            res = gp.run_displacement(ds, g, a.mode, dirs=dirs, make_plots=False)
            print(f"[{k}/{len(jobs)}] OK {ds} {g}  {time.time() - t0:.0f}s  "
                  f"-> {res.get('per_cell')}", flush=True)
        except Exception as e:
            print(f"[{k}/{len(jobs)}] FAIL {ds} {g}: "
                  f"{type(e).__name__}: {e}", flush=True)
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\n[all done] 总用时 {(time.time() - t_start) / 3600:.2f} 小时",
          flush=True)


if __name__ == "__main__":
    main()
