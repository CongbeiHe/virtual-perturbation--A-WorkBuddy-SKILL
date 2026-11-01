"""把用户级 virtual-perturbation SKILL 同步进 release 文件夹。

为什么需要转换
--------------
用户级 SKILL（`~/.workbuddy/skills/virtual-perturbation/`）里的路径是**本机绝对路径**
（`E:\\Geneformer\\venv\\...`），而 release 是要分发给别人的仓库副本，路径必须写成
`<ROOT>` 占位符。两边内容必须等价，否则 release 会逐渐过时
（2026-11-01 时发现 release 副本还停留在旧版，缺 4.1/4.2/5a/5b/第 10 节等）。

故本脚本做的事：读用户级 SKILL.md → 做路径/措辞泛化 → 写进 release。
**唯一真源是用户级 SKILL.md**，不要手改 release 里的副本（会被覆盖）。

用法
----
    E:\\Geneformer\\venv\\Scripts\\python.exe tools/sync_skill_to_release.py [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
SRC = Path(r"C:\Users\28115\.workbuddy\skills\virtual-perturbation")
DST = ROOT / "release" / "geneformer-pert" / "skills" / "virtual-perturbation"

# 顺序敏感：先替换长路径，再替换短路径
REPLACEMENTS = [
    # 环境路径 → 可移植占位
    # 注意：源文本里这些路径外层通常已有反引号（表格单元格），
    # 替换内容一律用**纯文本**，否则会出现反引号嵌套。
    (r"E:\Geneformer\venv\Scripts\python.exe", "<ROOT>/venv/Scripts/python.exe"),
    (r"E:\Geneformer\venv", "<ROOT>/venv"),
    (r"E:\Geneformer", "<ROOT>"),
    # 标题与措辞
    ("# Geneformer 虚拟扰动统一工作流（virtual-perturbation，用户级/跨工作空间）",
     "# Geneformer 虚拟扰动统一工作流（virtual-perturbation）"),
    ("本 Skill 定义在**任何工作空间**发起虚拟扰动分析时的固定方法、固定输出结构、固定可信度规则。",
     "本 Skill 定义在任何工作空间发起虚拟扰动分析时的固定方法、固定输出结构、固定可信度规则。  \n"
     "`<ROOT>` 指本仓库根目录（克隆后所在位置，下同）。"),
    ("环境探测：若 `GENEFORMER_HOME` 不存在或 venv 缺失，说明环境被移动过——先向用户确认新位置，"
     "再设 `GENEFORMER_HOME` 环境变量运行，不要盲目重装依赖。",
     "环境探测：若 venv 或模型缺失，提示用户运行 `setup_env` 与 `tools/download_model.py`，不要盲目重装。"),
    ("| 所有依赖（torch/scanpy/geneformer/geneformer_pert）都在此 venv |",
     "| 所有依赖（torch/scanpy/geneformer/geneformer_pert）都在此 venv；Linux 用 `venv/bin/python` |"),
    # 参照实现：明确标注是本机历史参照
    ("过表达 target⁺ + 零分布链（2026-11-01，归档在",
     "过表达 target⁺ + 零分布链（2026-11-01，本机归档在"),
    ("可直接复用的脚本实现（HIF3A 心肌项目，`E:\\HIF3A_virtual_cell\\scripts\\`）：",
     "可直接复用的脚本实现（HIF3A 心肌项目，`E:\\HIF3A_virtual_cell\\scripts\\`，"
     "**本机历史参照，不保证开箱即跑**）："),
]

# 需要一并同步的其它文件（内容一致，无需转换）
PLAIN_FILES = ["references/method_notes.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印将写入的目标与差异行数")
    a = ap.parse_args()

    if not (SRC / "SKILL.md").exists():
        sys.exit(f"源 SKILL.md 不存在: {SRC / 'SKILL.md'}")
    DST.mkdir(parents=True, exist_ok=True)

    text = (SRC / "SKILL.md").read_text(encoding="utf-8")
    n_sub = 0
    for old, new in REPLACEMENTS:
        if old in text:
            n_sub += text.count(old)
            text = text.replace(old, new)
        else:
            print(f"  [warn] 未命中，规则可能已失效: {old[:60]}...")

    # release 副本保留自己的 frontmatter description（分发版触发词不同）
    old_fm = (DST / "SKILL.md").read_text(encoding="utf-8") if (DST / "SKILL.md").exists() else None
    if old_fm and old_fm.startswith("---"):
        old_desc = old_fm.split("---")[1]
        if text.startswith("---"):
            parts = text.split("---")
            text = "---" + old_desc + "---" + "---".join(parts[2:])

    print(f"SKILL.md: {len(text.splitlines())} 行，应用替换 {n_sub} 处")
    if a.dry_run:
        print("--- dry-run，未写入 ---")
        return

    (DST / "SKILL.md").write_text(text, encoding="utf-8")
    print(f"  -> {DST / 'SKILL.md'}")

    for rel in PLAIN_FILES:
        s, d = SRC / rel, DST / rel
        if not s.exists():
            print(f"  [skip] 源不存在 {rel}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        print(f"  -> {d}")


if __name__ == "__main__":
    main()
