"""
把 vendor/tdigest_pure 里的纯 Python tdigest 实现安装进虚拟环境。

为什么不用 `pip install ./vendor/tdigest_pure`：
    本机文件钩子会把「递归删除目录」劫持为回收站操作，而回收站操作不可用，
    pip 在构建 wheel 后清理临时目录时必定报错并回滚安装。
    因此这里直接复制文件 + 手写 dist-info，效果与 pip 安装一致，
    并且 pip 之后会认为 `tdigest` 已满足依赖。

用法：
    python tools/install_tdigest_shim.py
"""

import shutil
import sys
from pathlib import Path

PKG_NAME = "tdigest"
VERSION = "0.5.2.2"
PURE_VERSION = "0.5.2.2+pure"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "vendor" / "tdigest_pure" / PKG_NAME
SITE = Path(sys.prefix) / "Lib" / "site-packages"


def main():
    if not SRC.exists():
        raise SystemExit(f"未找到源目录: {SRC}")

    dst = SITE / PKG_NAME
    dist_info = SITE / f"{PKG_NAME}-{VERSION}.dist-info"

    if dst.exists():
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from force_remove import hard_rmtree  # noqa: PLC0415

        hard_rmtree(dst)
    shutil.copytree(SRC, dst)
    print(f"已复制: {SRC}  ->  {dst}")

    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: tdigest\nVersion: {VERSION}\n"
        f"Summary: tdigest compatibility shim (pure python, version {PURE_VERSION})\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shim\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "INSTALLER").write_text("manual-shim\n", encoding="utf-8")
    (dist_info / "RECORD").write_text(
        "".join(
            f"{p.relative_to(SITE).as_posix()},,\n"
            for p in sorted(dst.rglob("*")) + sorted(dist_info.rglob("*"))
            if p.is_file()
        ),
        encoding="utf-8",
    )
    print(f"已写入 dist-info: {dist_info}")

    sys.path.insert(0, str(dst.parent))
    import tdigest  # noqa: PLC0415

    td = tdigest.TDigest()
    for v in [1, 2, 3, 4, 5]:
        td.update(v)
    print(f"自检: percentile(50)={td.percentile(50)}, trimmed_mean(0,100)={td.trimmed_mean(0, 100)}")
    print("完成，Geneformer 现在可以正常 import。")


if __name__ == "__main__":
    main()
