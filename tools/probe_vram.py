"""
显存压力探测：分档递增 batch size，逐步加压并实时监控显存。
用途：找出本机（RTX 5060 Laptop 8GB）在不触发驱动崩溃前提下的安全 batch size。

安全策略：
- 从极小 batch 开始（2），逐档递增
- 每档跑完立刻同步 + 清缓存
- 显存占用超过阈值（默认 80%）立即停止，不再往上探
- 每档之间留出间隔，避免驱动连续高压

用法：
    python tools/probe_vram.py --model models/V2-104M --max-batch 32
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def torch_dtype_from(name: str):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "V2-104M"))
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "tokenized" / "pbmc3k.dataset"))
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--safe-frac", type=float, default=0.80, help="显存占用超过该比例即停止加档")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else 0
    budget = total * args.safe_frac
    print(f"设备: {torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'}")
    print(f"总显存: {total/1024**3:.2f} GiB | 安全上限: {budget/1024**3:.2f} GiB ({args.safe_frac:.0%})")
    print(f"精度: {args.dtype} | 序列截断: {args.max_seq_len}\n")

    # 1) 加载模型
    dt = torch_dtype_from(args.dtype)
    model = AutoModelForMaskedLM.from_pretrained(args.model, torch_dtype=dt)
    model = model.to(dev).eval()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    print(f"[模型] 加载完成，常驻显存 {base/1024**3:.2f} GiB\n")

    # 2) 取样本
    ds = load_from_disk(args.data)
    print(f"[数据] {len(ds)} 个细胞，字段 {ds.column_names}")

    input_ids_key = "input_ids"
    lengths = [min(len(x), args.max_seq_len) for x in ds[input_ids_key][:2000:20]]
    print(f"[数据] 抽样序列长度: 中位 {sorted(lengths)[len(lengths)//2]}, 最大 {max(lengths)}\n")

    print(f"{'batch':>6} | {'峰值显存':>10} | {'增量':>8} | {'耗时':>8} | 状态")
    print("-" * 62)

    batch = 2
    results = []
    while batch <= args.max_batch:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            rows = [ds[i][input_ids_key][: args.max_seq_len] for i in range(min(batch, len(ds)))]
            maxlen = max(len(r) for r in rows)
            pad_id = 0
            ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((len(rows), maxlen), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
                att[i, : len(r)] = 1
            ids, att = ids.to(dev), att.to(dev)

            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
                hs = out.hidden_states[-1].mean(dim=1)
            torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000

            peak = torch.cuda.max_memory_allocated()
            delta = peak - base
            ratio = peak / total
            flag = "OK"
            if ratio > args.safe_frac:
                flag = "!! 接近上限，停止加档"
            print(f"{batch:>6} | {peak/1024**3:>8.2f}G | {delta/1024**3:>6.2f}G | {dt_ms:>6.0f}ms | {flag}")
            results.append((batch, peak, delta, dt_ms))

            del ids, att, out, hs
            torch.cuda.empty_cache()
            gc.collect()

            if ratio > args.safe_frac:
                break
            batch *= 2

        except torch.cuda.OutOfMemoryError as e:
            print(f"{batch:>6} | {'--':>10} | {'--':>8} | {'--':>8} | OOM（显存不足，安全退出）")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"{batch:>6} | {'--':>10} | {'--':>8} | {'--':>8} | 异常: {type(e).__name__}: {e}")
            break

    print("\n=== 结论 ===")
    if results:
        safe = [b for b, peak, d, t in results if peak / total <= 0.7]
        print(f"建议 batch size: <= {max(safe) if safe else results[0][0]}（显存占用 < 70%）")
        for b, peak, d, t in results:
            print(f"  batch={b:<4} 峰值={peak/1024**3:.2f}G ({peak/total:.0%})  单次={t:.0f}ms")
    else:
        print("未能完成任何档位测试")


if __name__ == "__main__":
    main()
