"""虚拟扰动封装包：方法学常量集中管理。

所有常量与 scripts/21-23 的已验证实验严格一致，改动任何一项都会破坏可复现性。

路径解析（跨工作空间核心）：
  - GENEFORMER_HOME        模型/代码/token 字典所在根目录（默认 E:\\Geneformer）
  - GENEFORMER_PERT_WORKDIR data/outputs 写入根目录（默认 = GENEFORMER_HOME）。
    在其他工作空间运行时可指向该工作空间，实现输出隔离；模型与代码始终取自 HOME。
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(os.environ.get("GENEFORMER_HOME", r"E:\Geneformer")).resolve()
WORKDIR = Path(os.environ.get("GENEFORMER_PERT_WORKDIR", str(ROOT))).resolve()
CODE_DIR = ROOT / "code" / "geneformer"
MODEL_DIR = ROOT / "models" / "V2-104M"
DATA_DIR = WORKDIR / "data"
OUTPUT_DIR = WORKDIR / "outputs"
TOKEN_DICT = CODE_DIR / "token_dictionary_gc104M.pkl"
GENE_NAME_ID_DICT = CODE_DIR / "gene_name_id_dict_gc104M.pkl"

# ---------------------------------------------------------------- 模型与方法学
MODEL_VERSION = "V2"
BATCH = 8                      # forward_batch_size（与 22/23 号实验一致）
EMB_LAYER = 0                  # ISP emb_layer
EMB_MODE = "cls_and_gene"      # 同时记录 cell_emb 与逐基因余弦
CELL_EMB_STYLE = "mean_pool"
NPROC = 8
VRAM_LIMIT_FRAC = 0.85

# UMAP（与 23 号一致；random_state 固定保证可复现）
UMAP_KW = dict(n_neighbors=15, min_dist=0.5, metric="cosine",
               random_state=42, n_jobs=1)

# 扰动序列构造（与库内 perturb_emb_by_rank 语义一致）：
#   1) 删除目标基因原 token
#   2) 插入到 <cls> 之后（位置 1）
#   3) 超长时截断：保留 <eos>，丢弃末位基因
PERTURB_INSERT_POS = 1

# 随机基线（方向凝聚度置换检验）
N_PERM = 1000
PERM_SEED = 42

# 基因过滤（Geneformer 映射）
MIN_NCOUNTS = 1                # 剔除零计数细胞

# 交叉验证容差
XVAL_TOL = 1e-3                # 与库内 ISP 排序后余弦最大差

# ---------------------------------------------------------------- 输出结构约定
def default_dirs(dataset_name: str, target_symbol: str, mode: str):
    """标准输出目录（与既有实验命名一致）。"""
    tag = f"{dataset_name}_{target_symbol.lower()}_{mode}"
    isp = OUTPUT_DIR / f"{dataset_name}_isp"
    return {
        "counts_dir": DATA_DIR / f"{dataset_name}_counts",
        "dataset": DATA_DIR / "tokenized" / f"{dataset_name}.dataset",
        "isp_raw": isp / "raw",
        "isp_gene": isp / "gene",
        "isp_cell": isp / "cell",
        "stats": OUTPUT_DIR / f"{dataset_name}_isp_stats",
        "analysis": OUTPUT_DIR / f"{dataset_name}_{target_symbol.lower()}_{mode}_analysis",
        "tag": tag,
    }


# ---------------------------------------------------------------- 支持的扰动模式
# 键 = 用户/Agent 可用的 mode 名；值 = InSilicoPerturber 的 perturb_type
PERTURB_MODES = {
    "overexpress": "overexpress",   # 过表达：删原 token，插到 <cls> 后
    # 敲除：删除该基因 token。⚠ 官方 InSilicoPerturber 不接受 "knockout"
    # （validate_options 只认 {'delete','inhibit','activate','overexpress'}），
    # 库内等价别名是 "delete"。mode 字符串仍用 "knockout"（用于输出目录命名），
    # 只有传给库的 perturb_type 换成 "delete"，语义完全等价。
    "knockout": "delete",
    "delete": "delete",             # 同 knockout 语义（库内别名）
    "activate": "activate",         # 激活：rank 前移 1
    "inhibit": "inhibit",           # 抑制：rank 后移 1
}
