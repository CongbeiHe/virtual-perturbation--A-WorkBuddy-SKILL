"""核查 normal_geneformer.h5ad 的组织/来源标签，确认组织归属。"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

H5AD = r"E:\Geneformer\data\tokenized\_input\normal_geneformer.h5ad"
VNN1 = "ENSG00000112299"

print("读取数据集 ...", flush=True)
a = ad.read_h5ad(H5AD)
obs = a.obs
print(f"shape: {a.shape[0]} cells x {a.shape[1]} genes\n")

LABEL_COLS = ["harm_study", "harm_healthy.tissue", "harm_tumor.site", "harm_sample.type",
              "harm_condition", "tissue", "tissue_type", "assay", "disease",
              "suspension_type", "donor_id"]

for c in LABEL_COLS:
    if c not in obs.columns:
        print(f"[缺失列] {c}")
        continue
    vc = obs[c].astype(str).value_counts()
    print(f"=== {c}  ({vc.shape[0]} 个取值) ===")
    print(vc.head(12).to_string())
    print()

print("=" * 70)
print("=== cell_type x tissue 交叉表 ===")
ct = pd.crosstab(obs["cell_type"].astype(str), obs["tissue"].astype(str))
print(ct.to_string())
print()

print("=== harm_study x tissue 交叉表 ===")
print(pd.crosstab(obs["harm_study"].astype(str), obs["tissue"].astype(str)).to_string())
print()

print("=== harm_study x cell_type 交叉表（前 8 类细胞）===")
top_ct = obs["cell_type"].astype(str).value_counts().head(8).index
sub = obs[obs["cell_type"].astype(str).isin(top_ct)]
print(pd.crosstab(sub["harm_study"].astype(str), sub["cell_type"].astype(str)).to_string())
print()

# ---- VNN1 表达是否随组织/研究变化 ----
X = a.X.tocsc() if sp.issparse(a.X) else a.X
idx = int(np.where(np.asarray(a.var.index == VNN1) |
                   np.asarray(a.var["ensembl_id"].astype(str) == VNN1))[0][0])
col = np.asarray(X[:, idx].todense()).ravel() if sp.issparse(X) else np.asarray(X[:, idx]).ravel()

df = pd.DataFrame({"tissue": obs["tissue"].astype(str).values,
                   "study": obs["harm_study"].astype(str).values,
                   "ct": obs["cell_type"].astype(str).values,
                   "v": col})
print("=== VNN1 表达 by tissue ===")
g = df.groupby("tissue").agg(n=("v", "size"), pos=("v", lambda s: int((s > 0).sum())),
                             mean=("v", "mean"), mx=("v", "max"))
g["pos_frac"] = g["pos"] / g["n"]
print(g.sort_values("pos", ascending=False).to_string())
print()
print("=== VNN1 表达 by harm_study ===")
g2 = df.groupby("study").agg(n=("v", "size"), pos=("v", lambda s: int((s > 0).sum())),
                             mean=("v", "mean"), mx=("v", "max"))
g2["pos_frac"] = g2["pos"] / g2["n"]
print(g2.sort_values("pos", ascending=False).to_string())
print()

# ---- 肝/肺谱系 marker 基因检测 ----
print("=== 谱系 marker 基因在数据集中的表达（判断组织类型）===")
g2i = None
import pickle
g2i = pickle.load(open(r"E:\Geneformer\code\geneformer\gene_name_id_dict_gc104M.pkl", "rb"))
MARKERS = {
    "肝-肝细胞": ["ALB", "APOA2", "TTR", "TF", "HP", "APOB", "CYP2E1", "ASGR1", "HNF4A"],
    "肝-胆管": ["KRT19", "KRT7", "SOX9", "EPCAM"],
    "肝-库普弗/星状": ["CLEC4F", "MARCO", "VSIG4", "GFAP", "ACTA2", "LRAT"],
    "肝-内皮(LSEC)": ["CLEC4G", "CLEC1B", "STAB2", "LYVE1", "FCN2"],
    "肺-AT1/AT2": ["AGER", "PDPN", "CAV1", "SFTPC", "SFTPA1", "SFTPB", "ABCA3"],
    "肺-气道上皮": ["SCGB1A1", "SCGB3A2", "KRT5", "FOXJ1", "MUC5AC", "TP63"],
    "肾": ["UMOD", "NPHS2", "SLC12A1", "PDZK1IP1"],
    "通用免疫": ["PTPRC", "CD3E", "MS4A1", "LYZ", "SPI1"],
    "VNN 家族": ["VNN1", "VNN2", "VNN3"],
}
var_ids = set(a.var.index.astype(str)) | set(a.var["ensembl_id"].astype(str))
for grp, genes in MARKERS.items():
    print(f"--- {grp} ---")
    for g in genes:
        eid = g2i.get(g)
        if not eid:
            print(f"   {g:<10} (字典无)")
            continue
        if eid not in var_ids:
            print(f"   {g:<10} (数据集无此基因)")
            continue
        j = int(np.where(np.asarray(a.var.index == eid) |
                         np.asarray(a.var["ensembl_id"].astype(str) == eid))[0][0])
        c2 = np.asarray(X[:, j].todense()).ravel() if sp.issparse(X) else np.asarray(X[:, j]).ravel()
        print(f"   {g:<10} {eid}  表达细胞 {int((c2>0).sum()):>7} ({(c2>0).mean():>6.2%})  "
              f"均值 {c2.mean():>8.3f}  最大 {c2.max():>6.0f}")
    print()
