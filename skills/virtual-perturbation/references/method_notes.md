# 方法学细节与历史实验索引

## 1. 已完成实验（同一方法、可直接引用对比）

| 实验 | 细胞集 | 目标/模式 | 输出目录 | 关键数字 |
|---|---|---|---|---|
| liver | 全肝细胞（GSE189175） | VNN1 OE | outputs/liver_isp* | 首次建链 |
| nashhcc | 全 malignant 3,844 | VNN1 OE | outputs/nashhcc_isp*, outputs/nashhcc_vnn1_analysis | 1-cos 1.45e-4，UMAP disp 0.351 |
| nashhcc tumor | Tumor & malignant 3,693 | VNN1 OE | outputs/nashhcc_tumor_isp*, outputs/nashhcc_tumor_vnn1_analysis | 1-cos 1.45e-4，disp 0.368 |
| S2 逐簇 | S2_hcc 3,424（4 亚簇） | VNN1 OE 逐簇 | outputs/nashhcc_s2_clusters | c2 凝聚度 0.72 vs 基线 0.33 |
| 真实 DE | Tumor & malignant 3,693 | VNN1+ (201) vs VNN1− | outputs/nashhcc_vnn1_real_de | 571 显著；99.3% 内部一致 |
| 模型校验 | 13,097 基因交集 | 幅度/方向 vs 真实 DE | outputs/nashhcc_vnn1_real_de/model_vs_real | 幅度 AUC 0.33；方向 57.4% |
| HIF3A target⁺ OE + 零分布（2026-11-01） | 心房 8,903 / 心室 10,243（HIF3A counts≥1） | HIF3A/MAP3K6/SLC10A6 OE | `E:\HIF3A_virtual_cell\targetpos_null_rank_mached\` | HIF3A 1-cos 4.968e-4/6.039e-4，交叉验证 2.4e-7/3.0e-7；4 对照 z=+2.35/+2.89（p=0.20）；MAP3K6/SLC10A6 全落在零分布内 |

## 2. 脚本 → 封装包对应关系

| 脚本 | 功能 | 封装位置 |
|---|---|---|
| 21_prepare_nashhcc_tumor.py | h5ad 子集准备 | geneformer_pert/prepare.py::prepare_subset |
| 22_nashhcc_tumor_vnn1_pipeline.py | tokenize+ISP+stats | geneformer_pert/pipeline.py::run_isp |
| 23_umap_nashhcc_tumor.py | UMAP 位移+图 | geneformer_pert/displacement.py::run_displacement |
| 24/26（凝聚度基线、逐细胞箭头） | 置换基线 | displacement.py::coherence_baseline |
| 27_vnn1_direction.py | 秩校准轴方向 | scripts/27（GPU 两遍前向，未封装；修正版未跑完） |
| 28-32 真实 DE 五链 + 校验 | 金标准锚定 | scripts/（依赖具体 h5ad，参数化价值低，按需复制改路径） |
| 33_symbol_strategy.py | 加符号策略评测 | scripts/33 |
| ~~13_null_control.py~~ | 旧对照法：随机稀有基因、**仅 3 个**、按表达细胞数（20~60）匹配 | **已过时，被 21/27 取代**：应改按 token rank 匹配、对照 ≥20 个、双字典校验 |
| 18/19（检出率 + rank 分布诊断） | 决定 target⁺ 限制是否值得做 | 未封装（CPU，纯诊断） |
| 12/20（target⁺ 子集 + 固定子样本） | 零分布配对用细胞集 | 未封装 |
| 21/27（挑 rank-matched 对照） | 零分布对照基因 | 未封装；**必须 token 词典 + 基因名字典双校验** |
| 22（零分布比较） | z / 经验 p / rank 校准残差 | 未封装 |
| 28（只跑 displacement 的批量驱动） | 零分布链加速 | 未封装，调 `geneformer_pert.displacement.run_displacement` |

## 3. 关键坑（新任务必读）

1. **gene_embs pickle 的列表覆盖全数据集细胞**（不只目标子集），顺序=按序列长度降序且仅含该基因的细胞。只映射子集会把所有值错位。
2. **cell_embs pickle 按长度降序保存**，与库内余弦对齐必须 `sorted(range(n), key=-len)`。
3. 截断边界细胞（len==4096 且不含目标基因）在库内/自写前向间必然不一致（截断语义差异），量级 ~1e-3，属已知。
4. 真实 DE 中 pct<2% 的极端稀疏基因 logFC 会出现 −23 这类假值，必须过滤。
5. scanpy `rank_genes_groups_df` 返回结果**已按 score 排序**，任何按列位置赋值都会错位——必须按基因名 merge。
6. pandas 列名 `shift` 会遮蔽 `DataFrame.shift` 方法，一律用 `df["shift"]`。
7. `bool` 稀疏矩阵的 `.mean(axis=0)` 不可靠，用 `.sum(axis=0)/n`。
8. **挑对照基因必须查两个字典**（2026-11-01 事故）：LINC02256 在 `gene_name_id_dict_gc104M.pkl` 里**有** ENSG，只在 `token_dictionary_gc104M.pkl` 里查无 token → `pipeline.py` L80 `tok[target_eid]` 抛 `KeyError`，两个 run 静默失败。**只校验前者会漏**。
9. **候选池默认太小**：`19_rank_match.py` 默认 `--top 10`（曾用 40），两个数据集交集可能只有 7 个，凑不出 20 个对照 → 用 `--top 300`。
10. **对照少时的高 z 是假象**：经验 p = (k+1)/(n+1)，n=4 时下限 0.20。HIF3A 4 对照 z=+2.35，加入第 5 个（PDZD2，1-cos 4.87e-4 逼近目标 5.00e-4）后掉到 +1.45。
11. **activate / inhibit 实际等价于 overexpress**：官方 `validate_options()` 直接 raise，`perturber_utils.py` 无实现；自研 `config.py` 那两行注释是空头承诺（`perturb_ids` 只用 `delete_only` 布尔分派）。
12. **归档移动时别动正在被写的文件**：链还在跑时移动 `data/`/`outputs/` 里的条目会崩链。同步脚本须先检查 `nvidia-smi` 无进程再执行。

## 4. 判读速查

- 1-cos 中位 ~1.5e-4 = 随机基因底噪级（CHRAC1 1.53e-4、MED12 1.68e-4），不得据此声称细胞态改变。
- 单组凝聚度显著性：coherence > 置换 p95（约 1/sqrt(n) 量级之上）。
- "定向迁移"成立还需：各组方向角互不相同（排除全局漂移）+ 幅度不低于其他组。
- 方向（UP/DOWN）只有在 A/B 级证据下才可写结论；C 级只写"模型预测幅度变化"。
- **基因特异性**只能由 rank-matched 零分布判定（≥20 对照），不由 1-cos 绝对值或凝聚度判定。
- 过表达先把不表达该基因的细胞剔掉（否则 1-cos 被抬高 8~11%），但**这本身不是显著性检验**。
- 目标基因 rank 中位 > ~500 时，限制到 target⁺ 基本无效（位移区间已覆盖整个头部），别浪费算力。

## 跨工作空间环境（v0.2.0 起）

- 包 geneformer_pert 已 editable 安装进 E:\Geneformer\venv（pyproject.toml 在 E:\Geneformer）
- 任何 cwd 均可 `E:\Geneformer\venv\Scripts\python.exe -m geneformer_pert ...`
- 路径解析：GENEFORMER_HOME（模型/代码，默认 E:\Geneformer）+ GENEFORMER_PERT_WORKDIR / --workdir（输出根，默认=HOME）
- prepare 已泛化：counts 层自动探测（layers['counts'] → 整数 X → 显式 --counts-layer）；var_names 自动识别符号/Ensembl（--gene-col 可显式指定）
- 冒烟验证（2026-10-28）：跨目录 CLI、workdir 切换、S3_hcc 6 细胞 prepare 全通过，映射数 35085 与历史一致
