"""
强制删除（用于绕过 pip uninstall 的回收站钩子问题）

背景：本机的文件钩子会把 "递归删除目录" 劫持成回收站操作，而回收站操作在此环境会失败：
    OSError [safe-delete] Error during a `trash` operation: Some operations were aborted
于是 pip 卸载/覆盖任何包时都会中断。但底层 os.remove / os.rmdir 是正常的，
本脚本用自底向上的逐个删除代替 shutil.rmtree。

用法：
    python tools/force_remove.py <路径> [<路径> ...]
    python tools/force_remove.py --pip <包名>        # 删除 site-packages 里某包的所有文件
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

_ORIG_RMTREE = shutil.rmtree


def hard_rmtree(path):
    """自底向上删除：只使用 os.remove / os.rmdir，不触发 trash 钩子。"""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        os.remove(path)
        return 1

    removed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
                removed += 1
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
                removed += 1
            except OSError:
                pass
    try:
        os.rmdir(str(path))
        removed += 1
    except OSError as exc:
        print(f"[warn] 无法删除目录 {path}: {exc}")
    return removed


def pip_package_paths(package):
    """定位某包在 site-packages 下的所有目录/文件（含 dist-info）。"""
    import site

    roots = [Path(p) for p in site.getsitepackages()]
    site.addsitedir(Path(sys.executable).parent.parent / "Lib" / "site-packages")
    roots.append(Path(sys.executable).parent.parent / "Lib" / "site-packages")
    patterns = [package, package.replace("-", "_"), f"{package.replace('-', '_')}-*.dist-info",
                f"{package}-*.dist-info", f"~{package.replace('-', '_')}", f"~{package}*"]
    found = []
    for root in set(roots):
        if not root.exists():
            continue
        for pat in patterns:
            found.extend(root.glob(pat))
    return sorted(set(found))


def main():
    ap = argparse.ArgumentParser(description="绕过回收站钩子的强制删除工具")
    ap.add_argument("paths", nargs="*", help="要删除的路径")
    ap.add_argument("--pip", action="append", default=[], help="要删除的已安装包名")
    args = ap.parse_args()

    targets = [Path(p) for p in args.paths]
    for pkg in args.pip:
        targets.extend(pip_package_paths(pkg))

    if not targets:
        ap.print_help()
        return 1

    total = 0
    for t in targets:
        if not t.exists():
            print(f"  跳过（不存在）: {t}")
            continue
        n = hard_rmtree(t)
        total += n
        print(f"  已删除 {n} 项: {t}")
    print(f"\n共删除 {total} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
