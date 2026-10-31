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

## 3. 关键坑（新任务必读）

1. **gene_embs pickle 的列表覆盖全数据集细胞**（不只目标子集），顺序=按序列长度降序且仅含该基因的细胞。只映射子集会把所有值错位。
2. **cell_embs pickle 按长度降序保存**，与库内余弦对齐必须 `sorted(range(n), key=-len)`。
3. 截断边界细胞（len==4096 且不含目标基因）在库内/自写前向间必然不一致（截断语义差异），量级 ~1e-3，属已知。
4. 真实 DE 中 pct<2% 的极端稀疏基因 logFC 会出现 −23 这类假值，必须过滤。
5. scanpy `rank_genes_groups_df` 返回结果**已按 score 排序**，任何按列位置赋值都会错位——必须按基因名 merge。
6. pandas 列名 `shift` 会遮蔽 `DataFrame.shift` 方法，一律用 `df["shift"]`。
7. `bool` 稀疏矩阵的 `.mean(axis=0)` 不可靠，用 `.sum(axis=0)/n`。

## 4. 判读速查

- 1-cos 中位 ~1.5e-4 = 随机基因底噪级（CHRAC1 1.53e-4、MED12 1.68e-4），不得据此声称细胞态改变。
- 单组凝聚度显著性：coherence > 置换 p95（约 1/sqrt(n) 量级之上）。
- "定向迁移"成立还需：各组方向角互不相同（排除全局漂移）+ 幅度不低于其他组。
- 方向（UP/DOWN）只有在 A/B 级证据下才可写结论；C 级只写"模型预测幅度变化"。
