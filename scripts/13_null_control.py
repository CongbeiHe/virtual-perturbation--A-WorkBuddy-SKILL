"""阴性对照：随机选 3 个与 VNN1 稀有度相近的基因做过表达，量化噪声底噪。

⚠ DEPRECATED (2026-11-01)
-------------------------
本脚本是**初版**对照法：随机挑 3 个稀有度相近的基因，只按「表达细胞数」匹配。
它有三个已知缺陷，已被 rank-matched 方法取代：

1. 只匹配表达细胞数，**不匹配 token rank** —— 而 1-cos 的机械成分主要由
   「改动区间覆盖多少高 rank token」决定，rank 差 100 就会显著改变位移量。
2. 仅 3 个对照 —— 经验 p = (k+1)/(n+1) 下限 0.25，不可能达到显著。
3. 不校验 token 词典 —— 挑到有 ENSG 但无 token 的基因会让 run 崩。

请改用 **40-47 号脚本**（rank-matched null，见 README「Rank-matched null
controls」与 SKILL.md §10）。本脚本保留仅为复现历史结果。
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "code"))

from datasets import load_from_disk  # noqa: E402
from geneformer_safe import VRAMGuard  # noqa: E402

VNN1 = "ENSG00000112299"
OUT_DIR = ROOT / "outputs" / "liver_isp_analysis"
BATCH = 8
N_NULL = 3
SEED = 42


def log(m=""):
    print(m, flush=True)


tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
cls_id, eos_id, pad_id = tok["<cls>"], tok["<eos>"], tok["<pad>"]

# ---- 选 null 基因：在 tumor 原始计数中表达细胞数 20~60（VNN1=39）----
import anndata as ad
import scipy.sparse as sp

a = ad.read_h5ad(ROOT / "data" / "liver_counts" / "tumor" / "liver_tumor.h5ad")
X = a.X.tocsc()
n_pos = np.asarray((X > 0).sum(axis=0)).ravel()
cand = np.where((n_pos >= 20) & (n_pos <= 60))[0]
rng = np.random.default_rng(SEED)
g2i = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
i2g = {v: k for k, v in g2i.items()}
picked = []
for j in rng.permutation(cand):
    eid = str(a.var_names[j])
    if eid == VNN1 or eid not in tok:
        continue
    picked.append((eid, int(n_pos[j])))
    if len(picked) == N_NULL:
        break
log("null 基因: " + ", ".join(f"{i2g.get(e, e)}({e}) 表达{n}细胞" for e, n in picked))

# ---- 前向 ----
from transformers import BertConfig, BertForMaskedLM
import torch

ds = load_from_disk(str(ROOT / "data" / "tokenized" / "liver_tumor.dataset"))
all_ids = ds["input_ids"]
n = len(ds)
cfg = BertConfig.from_pretrained(str(ROOT / "models" / "V2-104M"))
max_len = cfg.max_position_embeddings

model = BertForMaskedLM.from_pretrained(str(ROOT / "models" / "V2-104M"))
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


def perturb_ids(ids, gene_tok):
    ids = list(ids)
    if gene_tok in ids:
        ids.remove(gene_tok)
    ids.insert(1, gene_tok)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [ids[-1]]
    return ids


order = sorted(range(n), key=lambda i: -len(all_ids[i]))
res = {}
for eid, ncell in picked:
    gt = tok[eid]
    t0 = time.time()
    cos_list = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            o_seqs = [list(all_ids[i]) for i in idx]
            p_seqs = [perturb_ids(all_ids[i], gt) for i in idx]
            L = max(max(len(x) for x in o_seqs), max(len(x) for x in p_seqs))
            def enc(seqs):
                inp = torch.full((len(seqs), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                for r, seq in enumerate(seqs):
                    inp[r, : len(seq)] = torch.tensor(seq)
                    att[r, : len(seq)] = 1
                return inp.cuda(), att.cuda()
            oi, oa = enc(o_seqs)
            pi, pa = enc(p_seqs)
            with VRAMGuard(limit_frac=0.85, tag="null"):
                ho = model(input_ids=oi, attention_mask=oa, output_hidden_states=True).hidden_states[-1][:, 0, :]
                hp = model(input_ids=pi, attention_mask=pa, output_hidden_states=True).hidden_states[-1][:, 0, :]
            ho, hp = ho.float().cpu().numpy(), hp.float().cpu().numpy()
            c = np.sum(ho * hp, 1) / (np.linalg.norm(ho, axis=1) * np.linalg.norm(hp, axis=1) + 1e-12)
            for r, i in enumerate(idx):
                cos_list[i] = c[r]
            del oi, oa, pi, pa, ho, hp
            torch.cuda.empty_cache()
    res[eid] = cos_list
    log(f"{i2g.get(eid, eid)}: 用时 {time.time()-t0:.0f}s  "
        f"1-cos 中位 {np.median(1-cos_list):.3e}  p90 {np.percentile(1-cos_list,90):.3e}")

# VNN1 基准
z = np.load(OUT_DIR / "cls_embs.npz")
eo, ep = z["orig"].astype(np.float64), z["pert"].astype(np.float64)
cos_v = np.sum(eo * ep, 1) / (np.linalg.norm(eo, axis=1) * np.linalg.norm(ep, axis=1) + 1e-12)
res[VNN1] = cos_v

rows = []
for eid, c in res.items():
    rows.append({"gene": i2g.get(eid, eid), "ensembl": eid,
                 "n_cells_express": int(n_pos[list(map(str, a.var_names)).index(eid)]) if eid != VNN1 else 39,
                 "cos_median": float(np.median(c)),
                 "one_minus_cos_median": float(np.median(1 - c)),
                 "one_minus_cos_p90": float(np.percentile(1 - c, 90))})
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "null_control.csv", index=False)
log("")
log(df.to_string(index=False))
log("null 对照完成")
