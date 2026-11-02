---
name: virtual-perturbation
description: Geneformer 虚拟扰动统一工作流（跨工作空间、跨数据集）。当用户要求对任意单细胞数据集做虚拟过表达/敲除/激活/抑制某基因、分析下游基因位移、细胞态迁移、逐簇扰动响应、扰动方向推断、或与真实差异表达对照验证时使用。触发词：虚拟扰动、in silico perturbation、过表达、敲除、knockout、overexpress、扰动后细胞态、下游基因位移。
---

# Geneformer 虚拟扰动统一工作流（virtual-perturbation）

本 Skill 定义在任何工作空间发起虚拟扰动分析时的固定方法、固定输出结构、固定可信度规则。  
`<ROOT>` 指本仓库根目录（克隆后所在位置，下同）。所有封装代码在 `geneformer_pert` 包中（已 editable 安装进专用 venv，任何目录可直接调用）。改动方法学前必须先阅读"方法学红线"一节。

## 0. 固定环境（跨工作空间不变的部分）

| 项 | 路径 / 值 | 说明 |
|---|---|---|
| Python 解释器 | `<ROOT>/venv/Scripts/python.exe` | 所有依赖（torch/scanpy/geneformer/geneformer_pert）都在此 venv；Linux 用 `venv/bin/python` |
| 包调用 | `python -m geneformer_pert ...` | 已 editable 安装，**任何 cwd 可用** |
| `GENEFORMER_HOME` | `<ROOT>`（默认，无需设置） | 模型 `models/V2-104M`、代码 `code/geneformer`、token 字典都在此 |
| 输出根目录 | 默认 = GENEFORMER_HOME；用 `--workdir <目录>` 切换 | 在其他工作空间运行时把 `--workdir` 指向该工作空间，实现输出隔离且与历史结果同构 |

环境探测：若 venv 或模型缺失，提示用户运行 `setup_env` 与 `tools/download_model.py`，不要盲目重装。

## 1. 需求澄清（用户未说明时逐项确认）

1. **细胞集**：哪个 h5ad、哪个细胞子集（cell_filter 表达式，作用于 obs）
2. **目标基因**：符号或 Ensembl ID；是否在数据中有表达
3. **模式**：overexpress（默认）/ knockout / activate / inhibit
4. **分析深度**：
   - L1 主链路：扰动 + 幅度表 + 细胞态 UMAP 位移（必做）
   - L2 精细：子集内聚类 + 逐簇凝聚度与置换基线（发现定向迁移）
   - L3 方向：秩校准轴（弱，只能当假设，准确率 ~58-61%）
   - L4 锚定：真实 DE 对照（金标准，强烈推荐）

## 2. 新数据集适配检查单（拿到陌生 h5ad 必做，先探测再运行）

```python
# 探针（用 <ROOT>/venv/Scripts/python.exe 运行）：
import anndata as ad
a = ad.read_h5ad(path, backed="r")
print(a.shape, list(a.obs.columns), list(a.var.columns), list(a.layers.keys()))
print(a.obs.head(3))
```

逐项确认：
1. **counts 来源**：优先 `layers['counts']`；若无且 X 为整数则用 X；都不满足时让用户指定 `--counts-layer`。严禁用归一化/缩放后的矩阵。
   - **例外（已验证可行）**：若 X 是**每细胞常数缩放 + log1p** 的保序变换，可按 `counts = round(expm1(X) × n_counts / T)` 还原（T 取该数据集 n_counts 中位数），并用"还原值是否 100% 落在整数上（偏差 <0.01）"验证。判据： `sum(expm1(X))` 在每细胞内恒定。Geneformer 分词只依赖细胞内基因降序排名，保序变换下词元序列与原始 counts 完全一致，方法学等价。参照实现 `E:\HIF3A_virtual_cell\scripts\01_denormalize.py`。
2. **基因标识**：var_names 是符号还是 Ensembl（`ENSG` 开头占比 >50% 即 Ensembl）；是 Ensembl 时自动找 var 中的符号列，找不到则用 `--gene-col` 指定。映射率 <50% 时提醒用户（可能是 mouse 数据或其他物种，gc104M 是人类模型——需先做同源基因转换并明确告知用户局限）。
3. **cell_filter 列名**：从实际 obs 列构造，不要假设有 `malignant`/`Tissue`。
4. **分组列**（L2 凝聚度分组、L4 簇混杂检查用）：确认样本/患者/细胞类型列名。
5. **目标基因检出率**：prepare 输出会打印；若为 0 需确认符号/物种。
6. **规模预估**：GPU 时间约 5-7 分钟/千细胞（ISP）+ 同量级（位移前向），>2 万细胞时先告知用户预计时长。

## 3. 标准执行流程

```bash
# L1 一条命令跑完（幂等，中断后重跑自动跳过已完成步骤）
# 在任何目录下运行（建议绝对路径解释器）：
<ROOT>/venv/Scripts/python.exe -m geneformer_pert all \
  --src <h5ad 绝对路径> \
  --cell-filter "<pandas query, 作用于 obs>" \
  --dataset-name <数据集名> \
  --target <基因符号> \
  --mode overexpress \
  --workdir <当前工作空间目录，可选>
```

分步：`prepare` → `perturb`（GPU）→ `displacement`（GPU）。编程调用：

```python
import geneformer_pert as gp   # 已安装，直接 import

dirs = gp.default_dirs("myset", "VNN1", "overexpress")
gp.prepare_subset("path/to/src.h5ad", "cell_filter", dirs["counts_dir"],
                  "myset", target_gene="VNN1")
gp.run_isp("myset", "VNN1", "overexpress")
res = gp.run_displacement("myset", "VNN1", "overexpress")
# res["coherence"]: 分组凝聚度; res["baseline"]: 置换随机基线 p95
# 判定：coherence > baseline["p95"] 才可声称定向迁移
```

## 4. 标准输出结构（不得改名；`<root>` = GENEFORMER_HOME 或 --workdir）

```
<root>/data/<set>_counts/<set>.h5ad            # 子集（counts int32, var=Ensembl）
<root>/data/tokenized/<set>.dataset            # tokenized
<root>/outputs/<set>_isp/raw|gene|cell/        # ISP raw pickle
<root>/outputs/<set>_isp_stats/<set>_<gene>_<mode>_gene_shifts.csv   # 幅度表（官方）
<root>/outputs/<set>_isp_stats/<set>_<gene>_<mode>_cell_shift.csv
<root>/outputs/<set>_<gene>_<mode>_analysis/
    cls_embs.npz  umap_coords.npz
    per_cell_shift.csv            # 逐细胞 cos + UMAP 前后坐标 + 位移
    coherence_by_<col>.csv        # 分组凝聚度
    coherence_baseline.csv        # 置换随机基线
    downstream_genes_robust.csv   # 幅度参考表（N>=100）
    umap_manifold.png  umap_displacement_arrows.png
    full_gene_results.csv         # 全量基因表（见 4.1，固定输出）
    umap_expression.png           # 表达量 UMAP（见 4.2，固定输出）
    umap_migration.png            # 定向迁移 UMAP（见 4.2，固定输出）
    REPORT.md                     # 按第 8 节模板
<root>/outputs/<gene>_all_gene_results.xlsx      # 多数据集时汇总，每数据集一个 sheet
```

### 4.1 全量基因结果表 `full_gene_results.csv`（固定输出）

合并 L1 幅度 + L3 方向 + L4 三条真实 DE 链 + 可信度分级，**覆盖所有有任一证据的基因**（不得只给 top-N）。固定列：

| 列 | 内容 |
|---|---|
| `L1_shift_amplitude` / `L1_n_detections` | 模型受影响幅度（标注"非显著性"） |
| `L3_model_direction` / `L3_dir_z` / `L3_dir_q` | 秩校准轴方向（UP/DOWN/NS） |
| `DE_logFC` / `DE_padj` | 全子集 Wilcoxon |
| `DM_logFC` / `DM_padj` | 深度匹配 Wilcoxon |
| `PC_partial_r` / `PC_padj` | 连续偏相关（控深度） |
| `depth_matched_agree` / `partial_corr_agree` / `model_agree` | 各链是否与全子集 DE 同向 |
| `grade` | A / B / B- / C+ / NS / D（见第 6 节） |
| `final_direction` | A/B/B- → `UP`/`DOWN`；C+ → `unreliable(depth-confounded)`；仅模型证据 → `model_only:UP/DOWN`；其余 → `none` |

排序：按 grade（A→D）再按 DE_padj。目标基因自身的 logFC 是伪值（阴性组恒 0），表中保留但报告须加注说明。参照实现 `E:\HIF3A_virtual_cell\scripts\08_export_umap.py` 的 `build_gene_table`。

### 4.2 固定配图规范

**表达量 UMAP `umap_expression.png`**：原始流形散点，未表达细胞铺浅灰底（`#e3e3e3`），表达细胞用**灰→红**色阶（`LinearSegmentedColormap(["#d9d9d9","#f2b8b8","#e06666","#c0392b","#7b241c"])`）按 log1p(counts) 上色，高表达画在上层。默认面板：目标基因 + 一个关键下游基因（如本项目 PDK4）。

**定向迁移 UMAP `umap_migration.png`**（突出定向流动，不画全部细胞）：
- 左图：位移幅度热图，灰底 + 灰红色阶，`vmax` 截断到 p95 增强对比；
- 右图：UMAP 空间切 22×22 网格，逐格算平均位移向量作为局部主流方向（每格 ≥8 细胞才画）；深色粗箭头画**局部向量场**；红色细箭头只画**方向一致细胞**（个体方向与所在格主流方向 cos>0.6 且位移 > 中位数），数量超 900 时按位移从大到小截取。其余细胞一律不画箭头。

## 4.3 GPU 调度红线

**任何时刻只允许一个 GPU 进程**（ISP/位移/L3 前向都占数 GB 显存）。两个 CUDA 进程并发 → 显存超订溢出到主机内存 → 吞吐劣化 50 倍以上并可能触发电源/温度保护重启（RTX 5060 Laptop 8GB 实测发生过）。多数据集/多分析必须**串行排队**；CPU-only 步骤（L4 真实 DE）可与 GPU 作业错峰并行。运行中用 `nvidia-smi --query-compute-apps=pid` 确认独占。

## 5. 方法学红线（改动即破坏与历史结果的可比性）

1. **精度**：fp32，`forward_batch_size=8`，`emb_layer=0`，`emb_mode="cls_and_gene"`，`cell_emb_style="mean_pool"`。
2. **扰动序列语义**：删原 token → 插 `<cls>` 后位置 1 → 超长（>4096）截断时**保 `<eos>` 丢末位基因**。库内官方截断语义不同（丢 `<eos>` 保末位基因），满长细胞会与官方 raw pickle 系统性不一致——这是已知且可解释的差异，不要当 bug 修。
3. **UMAP**：`n_neighbors=15, min_dist=0.5, metric="cosine", random_state=42`，扰动后坐标必须 `transform` 到原始流形，不允许重训。
4. **交叉验证**：自写前向必须与官方 ISP 的 cell_emb 余弦按**排序后**最大差 < 1e-3。官方 raw pickle 按**序列长度降序**保存，直接按数据集顺序对齐是错的。
5. **随机基线**：凝聚度必须与簇内重配对置换基线（N_PERM=1000, seed=42）的 95 分位比较，不许只报绝对值。
5a. **knockout 的两个坑（2026-10-29 SH3BGR 实测）**：
   - 官方 InSilicoPerturber 不接受 perturb_type="knockout"（只认 delete/inhibit/activate/overexpress），库内等价别名是 "delete"。config.PERTURB_MODES 已把 "knockout" 映射到 "delete"，mode 字符串仅用于目录命名。
   - **模型方向与 DE 的可比性随扰动语义翻转**：DE_logFC 是"目标基因阳 vs 阴"的相关方向；overexpress 模拟"高表达"→同向即一致，knockout 模拟"无表达"→**反向才一致**。分级里 model_agree 不翻转会让 A/B 级系统性污染（心房修复前 A=8 全是假同向，修复后 A=10 全部 KO:DOWN 且 8/10 是肌节/Z 线基因，与文献吻合）。08/07 已实现 DOWN_MODES={knockout,delete,inhibit} 反号，final_direction 用 `KO:` 前缀标因果方向。
   - knockout 只对表达该基因的细胞有效（其余位移严格为 0）：L1/L3 应在 **目标基因+ 子集** 上跑，L4 在父集上做（需要阴性组），输出目录用 --out-name 对齐子集名。
   - L3 阳性对照在 knockout 下期望方向是 **DOWN**；若实测 UP（SH3BGR 心房 dir_z=+94）说明秩校准轴在该模式反号，该次 L3 全部方向只能当线索。
5b. **1-cos 与 UMAP 位移脱钩**：SH3BGR 敲除 1-cos 1.3-2.0e-4（低于底噪、三者最低）但 UMAP 位移 0.38-0.39（三者最高）→ UMAP 位移对任何序列改动都敏感，**不能当效应量**；小子集（<3000 细胞）画迁移图网格要放稀（08 已自适应 11×15×22）。
5c. **过表达必须先做 target⁺ 限制 + rank-matched 零分布**（2026-11-01 确立，见第 10 节）：
   - 不表达该基因的细胞走的是"只 insert 不删除 → 长度 +1"的机械捷径，1-cos 系统性偏高 8~11%，会把旧结果整体抬高。
   - 只限制到 target⁺ **不构成显著性检验**——必须配 rank-matched 对照基因零分布，且对照 **≥20 个**（n=4 时经验 p 下限 0.20，永远无法显著）。
   - 判定"基因特异性效应"**只能**靠零分布检验；凝聚度高只说明位移同向，不代表基因特异性。
6. **结果解读的固定结论**（已在 NASH-HCC 数据上用真实 DE 验证，不要推翻）：
   - 余弦幅度**不能**判别"基因是否真的改变"：对真实 DE 显著性 AUC≈0.31-0.33。
   - `N_Detections>=100` 筛选对真实 DE 富集仅 1.06×，无判别力。
   - 秩校准轴方向准确率约 57-61%，且 |dir_z| 大小与正确率无关；高表达方向混杂细胞状态轴。
   - "Geneformer 是编码器"：扰动后序列 rank 全体机械 +1，无"重新排序"输出；方向只能从 embedding 解码或真实 DE 获得。

## 6. 可信度分级与写结论规则

| 级别 | 定义 | 允许的说法 |
|---|---|---|
| A | 真实 DE 显著 + 深度匹配同向 + 连续偏相关同向 + 模型同向 | 可写入结论（"扰动后 X 基因上调/下调"） |
| B | 真实 DE 显著 + ≥2 条验证链同向 | 可写入结论，注明"真实 DE 支持" |
| B- | 真实 DE 显著 + 仅 1 条验证链同向 | 可写，注明证据较弱 |
| C+ | 仅全子集 Wilcoxon 显著、深度匹配/偏相关均不支持 | **不写方向**——多为深度混杂假象（HIF3A 项目实测数千个） |
| C | 仅模型证据（cos 幅度或 L3 方向） | 只能说"模型预测受影响"，方向须标注"模型推断，准确率~60%" |
| NS | 真实 DE 不显著 | 不写 |
| D | 低检出（N<100）、无真实 DE 信息或证据矛盾 | 不写 |

注意：L3 模型方向可能与真实 DE 系统性相反（HIF3A 项目实测 NPPA/RYR2 真实 UP 模型判 DOWN）——任何 C 级方向结论都必须带准确率警告。

## 7. L4 真实 DE 锚定与 L2 逐簇分析

**L4（扰动后必做）**：目标基因 UMI>=1 vs =0（同子集内），Wilcoxon+BH；阳性 <100 或 76%+ 单 UMI 时必须同时报深度匹配版与连续偏相关；簇混杂检查（阳性细胞亚簇分布不得显著偏斜）。参照实现 `<ROOT>\scripts\29_31`。

**L2**：`<cls>` embedding 上 PCA(50)→cosine→Leiden（res∈{0.3,0.5,0.8,1.2} 按 marker 清晰度选）；逐簇基因响应从官方 gene_embs pickle 按"长度降序处理序+仅含该基因细胞"映射（列表覆盖**全数据集**细胞——映射错会全错），均值须与官方表一致到 1e-6。参照实现 `<ROOT>\scripts\24_25`。

## 8. 报告模板

每份分析输出 `<set>_<gene>_<mode>_analysis/REPORT.md`，必含：
1. 数据与细胞集定义（h5ad 路径、cell_filter、n、按分组分布；若 X 非原始 counts，记录还原方法与验证结果）
2. 扰动参数（gene/mode/batch/精度）
3. 细胞态结论：1-cos 中位 vs 随机基因底噪（~1.5e-4）；凝聚度 vs 基线 p95
4. 基因级：幅度 top（标注"幅度参考，非显著性"）；可信度分级表（含分布统计）
5. 关键基因四链证据交叉表（目标通路与数据集标志基因，全量列出 DE/DM/PC/L3 四列，不受 top-N 截断影响）
6. 真实 DE 锚定结果（若做）
7. 明确的"不能说什么"清单（见第 5 节固定结论）

报告之外固定交付（见 4.1/4.2）：`full_gene_results.csv`、`umap_expression.png`、`umap_migration.png`；多数据集时另汇总 `<gene>_all_gene_results.xlsx`。

## 9. 完整方法学与历史实验索引

见 `references/method_notes.md`（历史实验参数、已踩过的 7 个坑、判读速查表）。

可直接复用的脚本实现（HIF3A 心肌项目，`E:\HIF3A_virtual_cell\scripts\`，**本机历史参照，不保证开箱即跑**）：
`01_denormalize.py`（log1p 数据还原 counts）、`02_run_perturb.py`（L1 驱动）、`03_real_de.py`（L4）、`04_subsample.py`（分层抽样）、`05_l2_clusters.py`（L2）、`06_l3_direction.py`（L3）、`07_report.py`（REPORT.md 生成）、`08_export_umap.py`（全量基因表 + 4.2 配图）。

过表达 target⁺ + 零分布链（2026-11-01，本机归档在 `E:\HIF3A_virtual_cell\targetpos_null_rank_mached\`）：
`18_probe_detect.py`（检出率探测）、`19_rank_match.py`（rank 分布诊断 + 候选对照，支持 `--top N` 扩池）、`12_make_pos_subset.py`（target⁺ 子集）、`20_make_null_subsample.py`（固定细胞子样本）、`21_pick_null_controls.py`（挑对照，含 token 词典校验）、`22_null_comparison.py`（零分布比较，含 z / 经验 p / rank 校准）、`27_pick_hif3a20.py`（扩到 20 个对照）、`28_run_hif3a20_displacement.py`（只跑 displacement 的批量驱动）。

## 10. 过表达的 target⁺ 限制与 rank-matched 零分布（2026-11-01 确立）

### 10.1 过表达有两种 token 语义，必须先区分

源码 `code/geneformer/perturber_utils.py::overexpress_tokens()`：

| 细胞是否表达该基因 | 实际操作 | 序列长度 | 位移范围 |
|---|---|---|---|
| **表达**（token 在序列里） | `delete_indices()` 原位删除 + `insert(1, token)` | **不变** | [1, p−1]，即 rank p → 1 的 rank 提升 |
| **不表达**（`perturb_index == [-100]`） | **只 insert 不删除** | **+1** | 全部 L 个 token |

后果：不表达该基因的细胞走"整条序列全位移"的机械捷径，1-cos 系统性偏高——HIF3A 实测把全体值抬高 **+11.4%（心房）/ +9.8%（心室）**；机制检验里所有对照基因的"未表达/表达"1-cos 比值都在 1.02~2.81。

### 10.2 rank 位置决定机械位移量 → 决定这条路值不值得走

改动区间是 [1, p−1]，p 越大位移越多。**动手前先算目标基因的 rank 中位**（脚本 19）：

| 目标基因 rank 中位 | 限制到 target⁺ 的效果 |
|---|---|
| p < ~500（只动头部，未覆盖整个 rank<500 头部） | **有效清理**，1-cos 会真的下降 |
| p ≈ 500 或更大（位移区间几乎正好覆盖整个头部） | **无效**：机械成分原封不动，只是把样本量砍没了 |

实测（心房 / 心室 15k）：HIF3A 192 / 200 → 有效；MAP3K6 481 / 503、SLC10A6 485 / 523 → 无效（子集只剩 367/198 与 67/98 个细胞，基本无统计效能）。

### 10.3 target⁺ 限制：能解决什么、不能解决什么

**能**：① 消除"未表达细胞走全位移捷径"的污染——这正是 ISP3 点名要修的 *lengthens its rank-value encoding* 假象；② 过表达自动升级为**官方原生语义**（与 `genes_to_perturb="all"` 走的 `overexpress_indices_special()` 是同一操作），零成本白拿；③ 所有细胞受到的位移同质，不再有"整条 L 位移 vs 头部 p−1 位移"的系统性价差。

**不能**：① 位移仍是机械的，1-cos 主要成分还是位置效应；② 混杂只是换了地方——从"表达 vs 不表达"之间转移到 target⁺ 子集**内部**（rank 50 与 rank 1500 的细胞 1-cos 天然不同）；③ 选择偏倚反而变重（target⁺ 按是否表达挑细胞 → 系统性深度更高）；④ **没有零假设**——所以必须配 rank-matched 零分布。

### 10.4 rank-matched 对照零分布（强制，不是可选项）

在同一批 target⁺ 细胞上，对 rank 位置匹配的对照基因跑**同一条 L1 链**，得到 1-cos 的零分布。

对照基因挑选判据：
1. **两个字典都要查**：`token_dictionary_gc104M.pkl` **且** `gene_name_id_dict_gc104M.pkl`。
   ⚠ 事故：LINC02256 在基因名字典里**有** ENSG，只在 token 词典里查无此 token → `pipeline.py` L80 `tok[target_eid]` 抛 `KeyError`，两个 run 全挂。只校验前者会漏。
2. 跨数据集时**两个数据集都要匹配**（保证跨组织可比）。
3. |Δrank| 尽量小（门槛 ≤60，实际取到 ≤45.5）；rank 跨度应**刻意横跨**目标基因 rank（含比它更靠后的对照 → 机械位移更大 → 检验偏保守）。
4. det_frac ≥ 0.3（比较时按 `express_<gene>` 列配对、剔除未表达细胞）；子样本 2000 细胞时 det=0.3 仍有 ~600 个表达细胞。
5. **对照个数 ≥ 20**。经验 p = (k+1)/(n+1)：n=4 时下限 0.20，**再显著也下不去**；n=20 时下限 1/21 ≈ 0.048。Kendiukhov 用了 61 个。
6. **候选池要够大**：默认 top-40 时两个数据集交集可能只有 7 个，凑不出 16 个 → 用 `19_rank_match.py --top 300` 扩池（纯 CPU）。

**实测教训**：4 个对照时 HIF3A z=+2.35（p=0.20）；加入第 5 个对照 PDZD2（1-cos 4.87e-4，逼近目标 5.00e-4）后 **z 掉到 +1.45**。基因间真实离散度远大于小样本估计——**对照少时的高 z 很可能是假象**。

### 10.5 执行流程

以下脚本是 HIF3A 心肌项目的参照实现（见 §9），在新数据集上按同样 5 步改写路径即可；核心 API 是 `geneformer_pert` 的 `prepare_subset` / `run_isp` / `run_displacement`。

```bash
# 1) 探测检出率 + rank 位置（CPU，先决定值不值得做）
python scripts/18_probe_detect.py
python scripts/19_rank_match.py --genes HIF3A --top 300

# 2) 建 target⁺ 子集（counts >= 1）并跑目标基因 L1（GPU，串行）
python scripts/12_make_pos_subset.py --target HIF3A
python scripts/02_run_perturb.py --dataset Atrial15k_hif3ap --target HIF3A \
       --mode overexpress --no-plots

# 3) 挑对照 → 4) 在同一批细胞上跑对照 L1 → 5) 比较
python scripts/27_pick_hif3a20.py
python scripts/28_run_hif3a20_displacement.py     # 只跑 displacement
python scripts/22_null_comparison.py
```

**加速：只跑 displacement 跳过 ISP。** `displacement.py` 只在交叉验证时 glob 读 ISP 的 raw pickle（L187），缺失则 `xval=None` 直接跳过，不影响 `per_cell_shift.csv`。2000 细胞子样本从 **14.5 → 6.5 分钟/个**。**代价**：这批 run 没有官方 ISP 对拍值（交叉验证容差 1e-3 那道保险），须保留少数几个完整 run 作锚点。

### 10.6 判读规则

- **z** =（目标 1-cos − 对照均值）/ 对照 sd；**经验 p** = (k+1)/(n+1)，k = 1-cos ≥ 目标的对照个数。
- **rank 校准检验**（补充）：20 个对照的 rank 不相等，而 1-cos 的机械成分随 rank 增大 → 对「对照 1-cos ~ 对照 rank」做 OLS，用目标 rank 处的预测值作基准比较残差，消掉残余 rank 差异。
- **p > 0.05 不等于"无生物学效应"**：可能是对照个数不够、target⁺ 样本量太小（MAP3K6 367 细胞、SLC10A6 67 细胞基本无解），或 rank 尾部位置效应。这类阴性**不能读作生物学阴性**。
- 凝聚度高（远超置换基线 p95）只说明**位移同向**，不代表基因特异性；后者只能靠零分布检验。

### 10.7 已知陷阱

- **activate / inhibit 不可用**：官方 `in_silico_perturber.py` L306-309 `validate_options()` 对这两者直接 `raise`（"currently under development"），`perturber_utils.py` 里根本没有对应实现。自研 `geneformer_pert/config.py` 注释声称"activate=rank 前移 1 / inhibit=rank 后移 1"是**空头承诺**——`displacement.py::perturb_ids()` 只用布尔 `delete_only` 分派，**activate/inhibit 实际等价于 overexpress**。
- **Q1 等价性（省钱）**：官方 docstring 说 activate 对 Q1（rank 最低四分位）基因的处理就是 *move to front*，与现行 overexpress **完全等价**。目标基因若落在 Q1（如 HIF3A rank 192，Q1 边界 1~315），即使日后实现修正语义也**无需重跑**。真正会变的只有 Q2/Q3 的基因。
- 1-cos 硬编码底噪 ~1.5e-4 量级，不能当效应量读。
- UMAP 位移对任何序列改动都敏感，不能当效应量。

### 10.8 文献支撑

- **ISP3 Platform**（bioRxiv 2026, doi 10.64898/2026.09.17.752497）：长度保持 rank 编辑策略原文 *"moves overexpressed genes that are **already expressed** to the front"*；明确指出插入未表达基因会 *"lengthen its rank-value encoding"*。
- **Kendiukhov**（*Comput Biol Chem* 2026;**124**(P1):109079, doi 10.1016/j.compbiolchem.2026.109079）：21 个基因的 Geneformer 删除扰动 + **expression-matched empirical null（61 个对照基因）** → **0/21 通过 FDR**，原文 *"recommends expression-matched nulls as a minimum standard"*。
- **Ahlmann-Eltze et al.** *Nat Methods* 2025;**22**(8):1657-1661：5 个基础模型 vs 4 个平凡基线，无一超越。
- Geneformer 官方 `in_silico_perturber_stats.py` 的 `vs_null` 模式（L465-521）就是同一思路（随机基因 null + Wilcoxon + BH FDR），**代码现成但未启用**。

**边界要说清**：文献的 expression-matched null 主要在 **deletion/KO** 上验证，迁到 OE 是基于"token rank ≡ 细胞内表达降序位次"的等价性推广；"匹配整个 rank 分布"、rank 校准检验、以及把 target⁺ 限制与零分布打包成一个流程，都是本项目的细化，不是文献原文。
