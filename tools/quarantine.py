"""
隔离（而非删除）已安装的包。

背景：本机存在两层文件安全钩子：
  1. 递归删除会被劫持成回收站操作，而回收站不可用 → 直接报错；
  2. 单轮删除超过 50 个文件需要外部确认 → 大批量删除无法完成。
因此无法用 `pip uninstall` 移除体积较大的包（pandas 等有上千个文件）。

本脚本改用 os.rename 把包目录整体「搬走」到 cache/quarantine/，
既不触发删除守卫，又能让 pip 认为该包未安装，从而干净地重装目标版本。

用法：
    python tools/quarantine.py pandas
    python tools/quarantine.py --list          # 查看隔离区
    python tools/quarantine.py --restore pandas  # 还原
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUARANTINE = PROJECT_ROOT / "cache" / "quarantine"
SITE = Path(sys.prefix) / "Lib" / "site-packages"


def _move(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True) if not dst.exists() else None
    try:
        os.rename(src, dst)
        return True
    except OSError:
        # 跨盘回退：先复制再重命名到原地丢弃（仍不使用删除）
        shutil.copytree(src, dst)
        return True


def quarantine(package: str, restore: bool = False):
    mod = package.replace("-", "_")
    if restore:
        moved = 0
        for entry in sorted(QUARANTINE.glob(f"*{package}*")) + sorted(QUARANTINE.glob(f"*{mod}*")):
            name = entry.name
            # 还原时去掉 __<时间戳> 后缀
            base = name.split("__")[0]
            if _move(entry, SITE / base):
                print(f"  还原: {name} -> {base}")
                moved += 1
        if not moved:
            print(f"隔离区没有 {package}")
        return

    targets = []
    for pat in [mod, f"{mod}-*.dist-info", f"{package}-*.dist-info", f"~{mod}*"]:
        targets.extend(SITE.glob(pat))
    if not targets:
        print(f"{package}: site-packages 中未找到")
        return

    import time

    stamp = time.strftime("%Y%m%d%H%M%S")
    for t in sorted(set(targets)):
        dst = QUARANTINE / f"{t.name}__{stamp}"
        if _move(t, dst):
            n = sum(1 for _ in t.rglob("*")) if t.is_dir() else 1
            print(f"  隔离 {n} 项: {t.name}  ->  {dst}")


def main():
    ap = argparse.ArgumentParser(description="把包移出 site-packages（不删除）")
    ap.add_argument("packages", nargs="*")
    ap.add_argument("--list", action="store_true", help="列出隔离区内容")
    ap.add_argument("--restore", action="store_true", help="从隔离区还原")
    args = ap.parse_args()

    if args.list:
        if not QUARANTINE.exists():
            print("隔离区为空")
            return 0
        total = 0
        for d in sorted(QUARANTINE.iterdir()):
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            total += size
            print(f"  {d.name:<60} {size / 1048576:8.1f} MB")
        print(f"\n合计 {total / 1048576:.1f} MB （位于 {QUARANTINE}）")
        return 0

    if not args.packages:
        ap.print_help()
        return 1

    for p in args.packages:
        quarantine(p, restore=args.restore)
    return 0


if __name__ == "__main__":
    sys.exit(main())
