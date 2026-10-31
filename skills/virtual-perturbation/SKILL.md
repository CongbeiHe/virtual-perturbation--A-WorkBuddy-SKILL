---
name: virtual-perturbation
description: Geneformer 虚拟扰动统一工作流（跨工作空间、跨数据集）。当用户要求对任意单细胞数据集做虚拟过表达/敲除/激活/抑制某基因、分析下游基因位移、细胞态迁移、逐簇扰动响应、扰动方向推断、或与真实差异表达对照验证时使用。触发词：虚拟扰动、in silico perturbation、过表达、敲除、knockout、overexpress、扰动后细胞态、下游基因位移。
---

# Geneformer 虚拟扰动统一工作流（virtual-perturbation）

本 Skill 定义在任何工作空间发起虚拟扰动分析时的固定方法、固定输出结构、固定可信度规则。  
`<ROOT>` 指本仓库根目录（克隆后所在位置，下同）。改动方法学前必须先阅读"方法学红线"。

## 0. 固定环境

| 项          | 路径 / 值                                                             |
| ---------- | ------------------------------------------------------------------ |
| Python 解释器 | `<ROOT>/venv/Scripts/python.exe`（Linux: `venv/bin/python`）         |
| 包调用        | `python -m geneformer_pert ...`（setup_env 已 editable 安装，任何 cwd 可用） |
| 模型         | `<ROOT>/models/V2-104M`（由 `tools/download_model.py` 下载）            |
| 输出根        | 默认 `<ROOT>`；其他工作空间用 `--workdir <目录>` 切换                            |

环境探测：若 venv 或模型缺失，提示用户运行 `setup_env` 与 `download_model.py`，不要盲目重装。

## 1. 需求澄清（用户未说明时逐项确认）

1. **细胞集**：哪个 h5ad、哪个细胞子集（cell_filter 作用于 obs）
2. **目标基因**：符号或 Ensembl ID；是否在数据中有表达
3. **模式**：overexpress（默认）/ knockout / activate / inhibit
4. **分析深度**：L1 主链路（必做）→ L2 逐簇精细 → L3 方向（弱，~58-61%）→ L4 真实 DE 锚定（金标准）

## 2. 新数据集适配检查单（陌生 h5ad 先探测再运行）

```python
import anndata as ad
a = ad.read_h5ad(path, backed="r")
print(a.shape, list(a.obs.columns), list(a.var.columns), list(a.layers.keys()))
```

1. counts 来源：`layers['counts']` → 整数 X → 显式 `--counts-layer`；严禁归一化矩阵
2. 基因标识：var_names 符号/Ensembl 自动识别（`--gene-col` 可显式指定）；  
   映射率 <50% 提醒用户（gc104M 是人类模型，非人类需同源转换并告知局限）
3. cell_filter 列名从实际 obs 构造，分组列确认
4. 目标基因检出率：prepare 输出打印；为 0 时查符号/物种
5. 规模预估：GPU 约 5-7 分钟/千细胞/阶段，>2 万细胞先告知时长

## 3. 标准执行流程

```bash
# L1 全链路（幂等）
<ROOT>/venv/Scripts/python.exe -m geneformer_pert all \
  --src <h5ad 绝对路径> --cell-filter "<query>" \
  --dataset-name <名> --target <基因> --mode overexpress [--workdir <输出根>]
```

编程调用：`import geneformer_pert as gp; gp.prepare_subset(...); gp.run_isp(...); gp.run_displacement(...)`

## 4. 标准输出结构（不得改名）

```
<root>/data/<set>_counts/<set>.h5ad
<root>/data/tokenized/<set>.dataset
<root>/outputs/<set>_isp/raw|gene|cell/
<root>/outputs/<set>_isp_stats/<set>_<gene>_<mode>_gene_shifts.csv
<root>/outputs/<set>_isp_stats/<set>_<gene>_<mode>_cell_shift.csv
<root>/outputs/<set>_<gene>_<mode>_analysis/
    cls_embs.npz  umap_coords.npz  per_cell_shift.csv
    coherence_by_<col>.csv  coherence_baseline.csv
    downstream_genes_robust.csv
    umap_manifold.png  umap_displacement_arrows.png  REPORT.md
```

## 5. 方法学红线（改动即破坏可复现性）

1. fp32，`forward_batch_size=8`，`emb_layer=0`，`emb_mode="cls_and_gene"`，`cell_emb_style="mean_pool"`
2. 扰动序列：删原 token → 插 `<cls>` 后位置 1 → 超长截断**保 `<eos>` 丢末位基因**；  
   官方 ISP 截断语义不同（丢 `<eos>` 保末位基因），满长细胞与官方 pickle 系统性不一致是已知差异
3. UMAP：`n_neighbors=15, min_dist=0.5, cosine, random_state=42`；扰动后坐标必须 `transform` 到原始流形
4. 交叉验证：自写前向与官方 cell_emb 余弦按**排序后**最大差 < 1e-3；  
   官方 raw pickle 按**序列长度降序**保存，按数据集顺序对齐是错的
5. 凝聚度必须与簇内重配对置换基线（N_PERM=1000, seed=42）95 分位比较
6. 固定结论（已用真实 DE 验证）：余弦幅度对真实 DE 显著性 AUC≈0.31-0.33（无判别力）；  
   N_Detections>=100 筛选富集仅 1.06×；秩校准轴方向 ~58-61% 且与自信度无关；  
   Geneformer 是编码器，扰动后 rank 全体机械 +1，无"重新排序"输出

## 6. 可信度分级

| 级别 | 定义                               | 允许的说法                        |
| -- | -------------------------------- | ---------------------------- |
| A  | 真实 DE 显著 + 深度匹配同向 + 偏相关同向 + 模型同向 | 可写入结论                        |
| B  | 三链一致（缺模型方向）                      | 可写入结论，注明"真实 DE 支持"           |
| C  | 仅模型证据                            | 只能说"模型预测受影响"；方向须标"模型推断，~60%" |
| D  | 低检出（N<100）或证据矛盾                  | 不写                           |

## 7. L4 / L2 参照实现

- L4 真实 DE：`scripts/29_31`（五链 DE + 一致性打分 + 幅度混杂分层）
- L2 逐簇：`scripts/24_25`（Leiden 聚类 + gene_embs pickle 逐簇映射）
- 注：scripts 为固定本机路径的历史参照，研究用，不保证开箱即跑

## 8. 报告模板

`<set>_<gene>_<mode>_analysis/REPORT.md` 必含：数据与细胞集定义、扰动参数、H  
细胞态结论（1-cos 中位 vs 随机底噪 ~1.5e-4；凝聚度 vs 基线 p95）、  
基因级幅度 top（标"幅度参考，非显著性"）、可信度分级表、  
真实 DE 锚定结果、"不能说什么"清单。

## 9. 方法学细节

见 `references/method_notes.md`。
