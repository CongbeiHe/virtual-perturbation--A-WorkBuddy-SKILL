"""下游基因结果深度分析：过滤低检测数基因，给出符号名与稳健排序。"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")
code = ROOT / "code" / "geneformer"

tok = pickle.load(open(code / "token_dictionary_gc104M.pkl", "rb"))
g2i = pickle.load(open(code / "gene_name_id_dict_gc104M.pkl", "rb"))
i2g = {v: k for k, v in g2i.items()}

df = pd.read_csv(ROOT / "outputs" / "liver_isp_stats" / "vnn1_oe_gene_shifts.csv")
print(f"总行数 {len(df)}")

df["affected_symbol"] = df["Affected_Ensembl_ID"].map(
    lambda e: i2g.get(e, np.nan) if isinstance(e, str) else np.nan)

# 排除 VNN1 自身
df = df[df["Affected_Ensembl_ID"] != "ENSG00000112299"].copy()

print(f"\n=== 检测次数分布 ===")
print(df["N_Detections"].describe().round(1).to_string())

# 稳健子集：至少 100 个细胞中检测到
rob = df[df["N_Detections"] >= 100].copy()
print(f"\n稳健基因（N_Detections>=100）: {len(rob)} 个")

rob["shift_down"] = 1 - rob["Cosine_sim_mean"]  # 表征变化幅度
top = rob.nlargest(40, "shift_down")
print("\n=== 过表达 VNN1 后表征位移最大的 40 个稳健基因 ===")
print(top[["affected_symbol", "Affected_Ensembl_ID", "Cosine_sim_mean",
           "Cosine_sim_stdev", "N_Detections", "shift_down"]].to_string(index=False))

# 位移最小的（最不受影响）
print("\n=== 受影响最小的 10 个稳健基因（对照）===")
print(rob.nsmallest(10, "shift_down")[["affected_symbol", "Cosine_sim_mean",
                                       "N_Detections", "shift_down"]].to_string(index=False))

# 已知相关基因的状态
print("\n=== 关键基因查询 ===")
queries = ["VNN1", "VNN2", "PPARG", "CD36", "SOD1", "SOD2", "NFE2L2", "KEAP1",
           "TXNRD1", "GPX4", "FOS", "JUN", "MYC", "AKT1", "MTOR", "TP53",
           "CTNNB1", "AXIN2", "HNF4A", "ALB", "AFP", "KRT19", "EPCAM", "TGFBI",
           "CXCL13", "ZIC2", "ASCL1", "CCN4", "FZD10", "IRX5", "OIT3", "GRIN2C"]
e2s = {v: k for k, v in g2i.items()}
for g in queries:
    eid = g2i.get(g)
    if eid is None:
        print(f"  {g:<10} (字典无)")
        continue
    row = df[df["Affected_Ensembl_ID"] == eid]
    if row.empty:
        print(f"  {g:<10} {eid}  (不在扰动结果中，可能无检测)")
    else:
        r = row.iloc[0]
        print(f"  {g:<10} {eid}  cos={r['Cosine_sim_mean']:.5f}  N={int(r['N_Detections']):>4}  shift={1-r['Cosine_sim_mean']:.5f}")

# 保存稳健表
rob.sort_values("shift_down", ascending=False).to_csv(
    ROOT / "outputs" / "liver_isp_analysis" / "downstream_genes_robust.csv", index=False)
top.to_csv(ROOT / "outputs" / "liver_isp_analysis" / "downstream_genes_top40.csv", index=False)
print("\n已保存: downstream_genes_robust.csv / downstream_genes_top40.csv")
