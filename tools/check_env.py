"""
环境自检：确认 PyTorch / CUDA / Geneformer 是否可用。

用法：
    python tools/check_env.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ok = True
    print("=" * 62)
    print(" Geneformer 本地环境自检")
    print("=" * 62)
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch

        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA 编译版本 : {torch.version.cuda}")
        avail = torch.cuda.is_available()
        print(f"CUDA 可用    : {avail}")
        if avail:
            cap = torch.cuda.get_device_capability(0)
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
            print(f"算力 (sm)    : sm_{cap[0]}{cap[1]}")
            props = torch.cuda.get_device_properties(0)
            print(f"显存         : {props.total_memory / 1024 ** 3:.1f} GiB")

            # 真正跑一次 GPU 矩阵乘，确认 kernel 与当前架构匹配
            try:
                a = torch.randn(2048, 2048, device="cuda")
                (a @ a).sum().item()
                print("GPU 运算测试  : 通过")
                del a
                torch.cuda.empty_cache()
            except Exception as exc:
                ok = False
                print(f"GPU 运算测试  : 失败 -> {exc}")
                print("  → 说明 PyTorch 的 CUDA kernel 不支持本机显卡架构，"
                      "需安装更高 CUDA 版本的 PyTorch（cu128 及以上）")
        else:
            ok = False
            print("  → 未检测到可用 GPU，将以 CPU 运行（会非常慢）")
    except ImportError:
        ok = False
        print("PyTorch     : 未安装")

    for mod in ["transformers", "datasets", "anndata", "scanpy", "numpy", "pandas"]:
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            ok = False
            print(f"{mod:<12}: 未安装")

    try:
        import geneformer

        print(f"{'geneformer':<12}: {Path(geneformer.__file__).parent}")
    except ImportError:
        ok = False
        print(f"{'geneformer':<12}: 未安装")

    for name in ["models/V2-104M", "models/V1-10M"]:
        p = PROJECT_ROOT / name
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"模型 {name:<14}: 存在 ({size / 1048576:.0f} MB)")
        else:
            print(f"模型 {name:<14}: 未下载")

    print("=" * 62)
    print("结果:", "全部就绪" if ok else "存在问题，请看上方 ✗ 项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
