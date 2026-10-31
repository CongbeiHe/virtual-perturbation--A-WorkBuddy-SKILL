# 一键部署脚本（Windows PowerShell）
# 用法: .\setup_env.ps1 [-Mirror]
param(
    [switch]$Mirror   # 国内网络加 -Mirror 使用 PyPI/HF 镜像
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python --version 2>$null
if (-not $?) { Write-Error "未找到 python，请先安装 Python 3.10-3.13 并加入 PATH" }

# 1. venv
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "[1/5] venv 已创建" -ForegroundColor Green
} else { Write-Host "[1/5] venv 已存在，跳过" -ForegroundColor Yellow }
$py = Join-Path $root "venv\Scripts\python.exe"

# 2. torch（CUDA 版）
& $py -c "import torch" 2>$null
if (-not $?) {
    if ($Mirror) { pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ }
    Write-Host "[2/5] 安装 PyTorch CUDA 12.8 版（50 系显卡；老显卡见 INSTALL.md 换 cu121）"
    & $py -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
} else { Write-Host "[2/5] torch 已安装，跳过" -ForegroundColor Yellow }

# 3. 其余依赖
Write-Host "[3/5] 安装 requirements.txt"
& $py -m pip install -r requirements.txt

# 4. editable 安装官方库与流水线包
Write-Host "[4/5] editable 安装 geneformer 与 geneformer_pert"
& $py -m pip install -e ./code --no-deps
& $py -m pip install -e . --no-deps

# 5. 验证
Write-Host "[5/5] 冒烟验证"
& $py -m geneformer_pert --help | Out-Null
if ($?) { Write-Host "部署完成。下一步: .\venv\Scripts\python.exe tools\download_model.py" -ForegroundColor Green }

# HF 镜像提示（国内网络）
if ($Mirror) {
    Write-Host "国内网络请在运行 download_model.py 前: " -NoNewline
    Write-Host '$env:HF_ENDPOINT = "https://hf-mirror.com"' -ForegroundColor Cyan
}
