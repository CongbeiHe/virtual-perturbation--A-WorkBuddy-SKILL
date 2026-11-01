"""
GSE189175 NASH-HCC malignant 细胞：VNN1 扰动的阴性对照（噪声底噪）。

随机挑选 2 个表达细胞数与 VNN1 相近（~195/3844）的基因做过表达，
用同一套前向流程量化 1-cos 的底噪范围。
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "code"))

from datasets import load_from_disk  # noqa: E402
from geneformer_safe import VRAMGuard, vram_report  # noqa: E402

VNN1 = "ENSG00000112299"
MODEL_DIR = ROOT / "models" / "V2-104M"
DS_DIR = ROOT / "data" / "tokenized" / "nashhcc_malignant.dataset"
SRC_H5AD = ROOT / "data" / "nashhcc_counts" / "malignant" / "GSE189175_malignant.h5ad"
OUT_DIR = ROOT / "outputs" / "nashhcc_vnn1_analysis"
BATCH = 8
N_NULL = 2
SEED = 42


def log(m=""):
    print(m, flush=True)


def main():
    t00 = time.time()
    tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    pad_id = tok["<pad>"]
    g2i = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    i2g = {}
    for name, eid in g2i.items():
        i2g.setdefault(eid, name)

    # ---- 选 null 基因：表达细胞数接近 VNN1 ----
    a = ad_read(SRC_H5AD)
    X = a.X.tocsc() if sp.issparse(a.X) else a.X
    n_pos = np.asarray((X > 0).sum(axis=0)).ravel()
    var_ids = np.array([str(v) for v in a.var_names])
    j_vnn1 = int(np.where(var_ids == VNN1)[0][0])
    target = int(n_pos[j_vnn1])
    log(f"VNN1 表达细胞数 {target}/{a.n_obs}")

    lo, hi = int(target * 0.85), int(target * 1.15)
    cand = [j for j in np.where((n_pos >= lo) & (n_pos <= hi))[0] if var_ids[j] != VNN1]
    rng = np.random.default_rng(SEED)
    picked = []
    for j in rng.permutation(cand):
        eid = var_ids[j]
        if eid in tok:
            picked.append((eid, int(n_pos[j])))
        if len(picked) == N_NULL:
            break
    log("null 基因: " + ", ".join(f"{i2g.get(e, e)}({e}) 表达{n}细胞" for e, n in picked))

    # ---- 前向 ----
    from transformers import BertConfig, BertForMaskedLM
    import torch

    cfg = BertConfig.from_pretrained(str(MODEL_DIR))
    max_len = cfg.max_position_embeddings
    ds = load_from_disk(str(DS_DIR))
    all_ids = ds["input_ids"]
    n = len(ds)
    log(f"细胞数 {n}  max_len {max_len}")

    info = vram_report()
    log(f"设备: {info['device']}  总显存 {info['total_gb']}G  batch={BATCH}")
    model = BertForMaskedLM.from_pretrained(str(MODEL_DIR))
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
                    ho = model(input_ids=oi, attention_mask=oa,
                               output_hidden_states=True).hidden_states[-1][:, 0, :]
                    hp = model(input_ids=pi, attention_mask=pa,
                               output_hidden_states=True).hidden_states[-1][:, 0, :]
                ho, hp = ho.float().cpu().numpy(), hp.float().cpu().numpy()
                c = np.sum(ho * hp, 1) / (np.linalg.norm(ho, axis=1) * np.linalg.norm(hp, axis=1) + 1e-12)
                for r, i in enumerate(idx):
                    cos_list[i] = c[r]
                del oi, oa, pi, pa, ho, hp
                torch.cuda.empty_cache()
                if (s // BATCH + 1) % 50 == 0:
                    log(f"  [{i2g.get(eid, eid)}] batch {s//BATCH+1}/{(n+BATCH-1)//BATCH}")
        res[eid] = cos_list
        log(f"{i2g.get(eid, eid)}: 用时 {time.time()-t0:.0f}s  1-cos 中位 {np.median(1-cos_list):.3e}  "
            f"p90 {np.percentile(1-cos_list, 90):.3e}")

    # VNN1 基准
    mine = pd.read_csv(OUT_DIR / "per_cell_shift.csv")
    res[VNN1] = mine["cos_orig_pert"].to_numpy()

    rows = []
    for eid, c in res.items():
        ncell = target if eid == VNN1 else dict(picked)[eid]
        rows.append({"gene": i2g.get(eid, eid), "ensembl": eid, "n_cells_express": int(ncell),
                     "cos_median": float(np.median(c)),
                     "one_minus_cos_median": float(np.median(1 - c)),
                     "one_minus_cos_p90": float(np.percentile(1 - c, 90))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "null_control.csv", index=False)
    log("")
    log(df.to_string(index=False))
    log(f"null 对照完成，总用时 {(time.time()-t00)/60:.1f} 分钟")


def ad_read(path):
    import anndata as ad
    return ad.read_h5ad(path)


if __name__ == "__main__":
    main()
