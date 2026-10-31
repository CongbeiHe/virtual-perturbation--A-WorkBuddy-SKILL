# 部署指南（Windows 优先，Linux 类似）

本仓库不含模型权重、数据集与虚拟环境（体积与许可原因），全部通过脚本获取。
按顺序执行即可完成部署，全程约 20-40 分钟（取决于网速）。

## 0. 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA + CUDA（本机验证环境：RTX 5060 Laptop 8GB，Blackwell `sm_120` 需 cu128） |
| Python | 3.10 - 3.13（验证环境 3.13） |
| Git | 拉取本仓库 |
| 网络 | 能访问 PyPI 与 HuggingFace（国内网络见下方镜像说明） |

## 1. 创建环境（setup_env.ps1 / setup_env.sh 自动完成）

```powershell
cd geneformer-pert
.\setup_env.ps1              # Git Bash: bash setup_env.sh
```

脚本做的事：
1. `python -m venv venv`
2. 安装 PyTorch（CUDA 版，见下方说明）
3. `pip install -r requirements.txt`
4. editable 安装两份代码：官方 geneformer（`code/`）与流水线包（`geneformer_pert/`）
5. 冒烟验证：`python -m geneformer_pert --help`

**PyTorch 版本选择**（关键）：

- RTX 50 系（Blackwell，`sm_120`）：必须 cu128+
  `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`
- RTX 30/40 系等较老卡：cu121/cu124 均可
  `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121`
- 无 N 卡：本流水线 GPU 步骤不可运行（prepare 仍可用）

**国内网络镜像**（可选，setup 脚本支持 `--mirror` 参数）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"          # HuggingFace 镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 2. 下载模型权重（约 400 MB，不入仓库）

```powershell
.\venv\Scripts\python.exe tools\download_model.py
```

从 HuggingFace 官方仓库 `ctheodoris/Geneformer` 拉取 V2-104M 到 `models/V2-104M/`。
权重许可见 HF 模型页（研究用途，禁止再分发——因此本仓库不携带权重文件）。

## 3. 准备测试数据（可选，375 MB）

方式一：到本项目的 **GitHub Releases** 页下载 `GSE189175_NASHHCC.h5ad`
（GEO: GSE189175 处理版，39,995 细胞 × 56,352 基因，含 `layers['counts']` 整数 UMI），放到 `task/GSE189175/`。

方式二：用自己的 h5ad。要求：

- 存在整数 counts 矩阵：`layers['counts']`，或 X 本身为整数；
- 基因标识为 HGNC 符号（var_names 或 var 某列）或 Ensembl ID；
- 非人类数据需先做同源基因转换（token 字典为人类 gc104M）。

数据集结构探测（只读）：

```python
import anndata as ad
a = ad.read_h5ad("your.h5ad", backed="r")
print(a.shape, list(a.obs.columns), list(a.layers.keys()), list(a.var.columns))
```

## 4. 冒烟测试（S3_hcc 子集，6 细胞，CPU 即可）

```powershell
.\venv\Scripts\python.exe -m geneformer_pert prepare `
  --src task\GSE189175\GSE189175_NASHHCC.h5ad `
  --cell-filter "malignant and Tissue=='Tumor' and SampleID=='S3_hcc'" `
  --dataset-name smoke --target VNN1
```

预期日志出现 `基因符号可映射到 gc104M: 35085 / 56352`。

## 5. 完整运行

```powershell
.\venv\Scripts\python.exe -m geneformer_pert all `
  --src task\GSE189175\GSE189175_NASHHCC.h5ad `
  --cell-filter "malignant and Tissue=='Tumor'" `
  --dataset-name nashhcc_tumor --target VNN1 --mode overexpress
```

参考耗时（RTX 5060 Laptop 8GB）：3,693 细胞 → prepare 数分钟、扰动 ~23 分钟、
位移分析 ~22 分钟。输出见 `outputs/nashhcc_tumor_vnn1_overexpress_analysis/`。

## 6. 常见问题

- **CUDA out of memory**：降 `--batch`（默认 8）；包内 vram-guard 会在显存吃紧时自动预警。
- **`no kernel image`**：torch CUDA 版本与显卡算力不匹配（50 系必须 cu128）。
- **tokenize 报 counts 非整数**：h5ad 用了归一化矩阵，请用 `--counts-layer` 指定原始计数层。
- **中断续跑**：直接重跑同一命令，已完成步骤自动跳过。
- **换输出位置**：加 `--workdir <目录>`（默认输出在仓库根的 `data/` 与 `outputs/`）。
