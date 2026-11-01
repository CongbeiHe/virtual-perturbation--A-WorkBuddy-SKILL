"""把 E:/Geneformer 的源码同步进 release/geneformer-pert。

与 `sync_skill_to_release.py` 的分工
------------------------------------
- 本脚本同步**代码与工程文件**（包、scripts、tools、vendored 官方源码、配置）。
  这些文件在两处应逐字节相同，直接复制即可。
- SKILL.md / method_notes.md **不由本脚本处理**：release 那份是刻意的可移植
  变体（`<ROOT>` 占位），必须走 `sync_skill_to_release.py` 做路径泛化后写入，
  否则会把本机绝对路径写进分发包。

安全规则
--------
- **只从源复制到 release，从不删除** release 里的文件。release 独有的东西
  （如 40-47 号脚本、README/INSTALL 等文档）不会被清掉。
- README.md 默认**不同步**：源那份是本机说明，release 那份是分发说明，
  两者内容刻意不同（见下方 EXCLUDE）。

用法
----
  python tools/sync_release.py [--dry-run] [--include-readme]
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(r"E:/Geneformer")
DST = ROOT / "release" / "geneformer-pert"

# (源目录, release 目录, 文件名 glob 列表)
DIR_RULES = [
    (ROOT / "geneformer_pert", DST / "geneformer_pert", ["*.py"]),
    (ROOT / "scripts", DST / "scripts", ["*.py", "*.ps1", "*.cmd"]),
    (ROOT / "tools", DST / "tools", ["*.py"]),
    (ROOT / "code" / "geneformer", DST / "code" / "geneformer", ["*.py", "*.pkl"]),
    (ROOT / "code" / "geneformer" / "mtl",
     DST / "code" / "geneformer" / "mtl", ["*.py"]),
    (ROOT / "code" / "geneformer" / "gene_dictionaries_30m",
     DST / "code" / "geneformer" / "gene_dictionaries_30m", ["*.pkl"]),
]

# 单文件：(源, 目标)
FILE_RULES = [
    (ROOT / "pyproject.toml", DST / "pyproject.toml"),
    (ROOT / "requirements.txt", DST / "requirements.txt"),
    (ROOT / "code" / "requirements.txt", DST / "code" / "requirements.txt"),
    (ROOT / "code" / "setup.py", DST / "code" / "setup.py"),
    (ROOT / "code" / "MANIFEST.in", DST / "code" / "MANIFEST.in"),
]

# release 独有、源里没有或不该被源覆盖的
EXCLUDE = {"README.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-readme", action="store_true",
                    help="连源 README.md 也同步（会覆盖分发说明，慎用）")
    a = ap.parse_args()

    if not DST.exists():
        sys.exit(f"release 目录不存在: {DST}")

    n_same = n_copy = n_skip = 0
    copied = []

    for src_dir, dst_dir, globs in DIR_RULES:
        if not src_dir.exists():
            print(f"[skip] 源目录不存在: {src_dir}")
            n_skip += 1
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for g in globs:
            for s in sorted(src_dir.glob(g)):
                if s.name in EXCLUDE and not a.include_readme:
                    continue
                d = dst_dir / s.name
                if d.exists() and filecmp.cmp(s, d, shallow=False):
                    n_same += 1
                    continue
                n_copy += 1
                copied.append(f"{s.relative_to(ROOT)} -> release/geneformer-pert/"
                              f"{d.relative_to(DST)}")
                if not a.dry_run:
                    shutil.copy2(s, d)

    for s, d in FILE_RULES:
        if not s.exists():
            print(f"[skip] 源文件不存在: {s}")
            n_skip += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists() and filecmp.cmp(s, d, shallow=False):
            n_same += 1
            continue
        n_copy += 1
        copied.append(f"{s.relative_to(ROOT)} -> release/geneformer-pert/"
                      f"{d.relative_to(DST)}")
        if not a.dry_run:
            shutil.copy2(s, d)

    print(f"\n相同 {n_same} 项 | 需更新 {n_copy} 项 | 跳过 {n_skip} 项")
    for c in copied:
        print(f"  {'[would] ' if a.dry_run else '[copy] '}{c}")
    if n_copy == 0:
        print("  release 已是最新，无需改动")
    if a.dry_run and n_copy:
        print("\n（--dry-run：未实际写入）")

    # 提醒 release 独有文件
    print("\n注：release 独有的文件不会被本脚本删除，例如：")
    for p in ["scripts/_ranknull.py", "scripts/40_probe_target_detection.py",
              "scripts/46_null_comparison.py", "README.md", "INSTALL.md"]:
        print(f"  - {p}")
    print("  SKILL.md / method_notes.md 请用 tools/sync_skill_to_release.py 同步")


if __name__ == "__main__":
    main()
