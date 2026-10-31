#!/usr/bin/env bash
# 一键部署脚本（Linux / Git Bash）
# 用法: bash setup_env.sh [--mirror]
set -e
cd "$(dirname "$0")"
MIRROR=false
[ "$1" == "--mirror" ] && MIRROR=true

command -v python >/dev/null || { echo "未找到 python，请先安装 Python 3.10-3.13"; exit 1; }

# 1. venv
if [ ! -d venv ]; then python -m venv venv; echo "[1/5] venv 已创建"; else echo "[1/5] venv 已存在，跳过"; fi
PY=venv/bin/python; [ "$(uname -s)" == "MINGW"* ] && PY=venv/Scripts/python.exe

# 2. torch CUDA
if ! "$PY" -c "import torch" 2>/dev/null; then
  [ "$MIRROR" = true ] && pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  echo "[2/5] 安装 PyTorch CUDA 12.8"
  "$PY" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
else echo "[2/5] torch 已安装，跳过"; fi

# 3. 依赖
echo "[3/5] 安装 requirements.txt"; "$PY" -m pip install -r requirements.txt

# 4. editable 安装
echo "[4/5] editable 安装 geneformer 与 geneformer_pert"
"$PY" -m pip install -e ./code --no-deps
"$PY" -m pip install -e . --no-deps

# 5. 验证
echo "[5/5] 冒烟验证"; "$PY" -m geneformer_pert --help >/dev/null \
  && echo "部署完成。下一步: $PY tools/download_model.py"
[ "$MIRROR" = true ] && echo '国内网络请先: export HF_ENDPOINT="https://hf-mirror.com"'
