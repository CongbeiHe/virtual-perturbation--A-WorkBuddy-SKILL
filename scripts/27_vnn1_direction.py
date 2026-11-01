"""
VNN1 虚拟过表达后，下游基因表达方向的判定（升/降）。

原理：
  ISP 的余弦相似度只有幅度没有方向。Geneformer 输入 token 按表达量排序，
  同一基因 token 在高表达/低表达细胞中的上下文 embedding 系统性不同。
  因此在未扰动状态下，用「该基因自身表达 rank 靠前的细胞的 embedding 均值
  − rank 靠后的均值」定义该基因的「上调方向向量」d_g；
  再把扰动引起的位移 (emb_pert − emb_orig) 投影到 d_g 上：
  投影为正 → 表达上升；为负 → 表达下降。

  阴性对照：把校准分组标签随机打乱重建 d_null，投影均值应≈0；
  阳性对照：VNN1 自身被强制移到 rank 1（最高表达位），其方向应显著为正。
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "code"))

import geneformer.perturber_utils as pu  # noqa: E402

VNN1 = "ENSG00000112299"
MODEL = ROOT / "models" / "V2-104M"
DS_PATH = ROOT / "data" / "tokenized" / "nashhcc_tumor_mal.dataset"
ROBUST_CSV = ROOT / "outputs" / "nashhcc_tumor_vnn1_analysis" / "downstream_genes_robust.csv"
OUT_DIR = ROOT / "outputs" / "nashhcc_tumor_vnn1_analysis"
BATCH = 8
MAX_LEN = 4096
TOP_K = 250
MIN_GROUP = 30
MIN_GROUP_VNN1 = 20
KEY_GENES = ["KEAP1", "HDAC3", "RELA", "INSIG2", "TXNRD1", "AKR1C3", "CPVL",
             "CYP2E1", "RBP4", "CYP3A5", "CYP2C8", "UGT2B11", "SLC30A10",
             "ALB", "SPP1", "FN1", "FOXA1", "DHRS2", "PLK1", "PFDN5"]


def log(msg=""):
    print(msg, flush=True)


def main():
    t00 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备: {device}")

    tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    gn = pickle.load(open(ROOT / "code" / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    pad_id = tok["<pad>"]
    vnn1_tok = tok[VNN1]

    # ---------- 候选基因 ----------
    robust = pd.read_csv(ROBUST_CSV)
    top = robust.nlargest(TOP_K, "shift_down")
    cand_names = (set(top["Affected_gene_name"]) | set(KEY_GENES)) - {"VNN1"}
    cand = {}
    for g in sorted(cand_names):
        eid = gn.get(g)
        if eid and eid in tok:
            cand[tok[eid]] = g
    cand[vnn1_tok] = "VNN1"          # 阳性对照
    log(f"候选基因 {len(cand)} 个")

    # ---------- 数据 ----------
    from datasets import load_from_disk
    ds = load_from_disk(str(DS_PATH))
    n = len(ds)
    order = sorted(range(n), key=lambda i: -len(ds[i]["input_ids"]))
    ids_all = ds["input_ids"]
    log(f"细胞 {n} 个，按长度降序处理")

    # ---------- 预计算：每候选基因在每细胞中的位置（rank 即位置） ----------
    cand_set = set(cand)
    detections = {t: [] for t in cand}      # token -> [(order_pos, orig_idx, position)]
    for oi, i in enumerate(order):
        for pos, t in enumerate(ids_all[i]):
            if t in cand_set:
                detections[t].append((oi, i, pos))
    n_det = {t: len(v) for t, v in detections.items()}
    log(f"检出数: 中位 {int(np.median(list(n_det.values())))}, "
        f"范围 [{min(n_det.values())}, {max(n_det.values())}]")

    # ---------- 校准分组 ----------
    hi_map, lo_map, all_map = {}, {}, {}    # token -> {orig_idx: position}
    for t, lst in detections.items():
        ranks = np.array([p for _, _, p in lst])
        q25, q75 = np.quantile(ranks, 0.25), np.quantile(ranks, 0.75)
        min_g = MIN_GROUP_VNN1 if t == vnn1_tok else MIN_GROUP
        hi = {(i, p) for oi, i, p in lst if p <= q25}
        lo = {(i, p) for oi, i, p in lst if p >= q75}
        hi_map[t] = {i: p for i, p in hi} if len(hi) >= min_g else {}
        lo_map[t] = {i: p for i, p in lo} if len(lo) >= min_g else {}
        all_map[t] = {i: p for _, i, p in lst}
    n_ok = sum(1 for t in cand if hi_map[t] and lo_map[t])
    log(f"可校准基因: {n_ok}/{len(cand)}")

    # ---------- 模型 ----------
    model = pu.load_model("Pretrained", 0, str(MODEL), mode="eval")
    layer_to_quant = pu.quant_layers(model) + 0
    dim = pu.get_model_emb_dims(model)
    log(f"hidden_states[{layer_to_quant}]  dim={dim}")

    emb_store = {t: {} for t in cand}       # token -> {orig_idx: np.float32[dim]}

    def run_batches(get_chunk, on_emb):
        """通用批前向；on_emb(batch_embs, chunk)"""
        for b0 in range(0, n, BATCH):
            chunk = get_chunk(b0)
            if not chunk:
                continue
            max_len = max(c[2] for c in chunk)
            tensors = [torch.tensor(c[1]) for c in chunk]
            padded = pu.pad_tensor_list(tensors, max_len, pad_id, MAX_LEN).to(device)
            amask = torch.tensor(
                [[1] * c[2] + [0] * (max_len - c[2]) for c in chunk], device=device)
            with torch.no_grad():
                outputs = model(input_ids=padded, attention_mask=amask)
            embs = outputs.hidden_states[layer_to_quant].cpu().numpy()
            del outputs
            on_emb(embs, chunk)
            if (b0 // BATCH) % 60 == 0:
                log(f"  {b0}/{n}  累计 {time.time()-t00:.0f}s")

    # ---------- Pass 1：orig 前向，存储 embedding + 校准累加 ----------
    log("Pass 1/2：未扰动前向 ...")
    seq_list = [(i, ids_all[i], len(ids_all[i])) for oi, i in enumerate(order)]
    acc = {t: {"hi": np.zeros(dim), "hn": 0, "lo": np.zeros(dim), "ln": 0} for t in cand}

    def on_orig(embs, chunk):
        for bi, (i, seq, L) in enumerate(chunk):
            for t, pm in all_map.items():
                p = pm.get(i)
                if p is None:
                    continue
                e = embs[bi, p, :].astype(np.float32)
                emb_store[t][i] = e
                hm, lm = hi_map[t], lo_map[t]
                if i in hm:
                    acc[t]["hi"] += e; acc[t]["hn"] += 1
                elif i in lm:
                    acc[t]["lo"] += e; acc[t]["ln"] += 1

    t0 = time.time()
    run_batches(lambda b0: seq_list[b0:b0 + BATCH], on_orig)
    log(f"Pass 1 完成，用时 {time.time()-t0:.0f}s")

    # ---------- 方向向量 ----------
    rng = np.random.default_rng(0)
    dvec, dnull, calib_n = {}, {}, {}
    for t, g in cand.items():
        hm, lm = hi_map[t], lo_map[t]
        if not (hm and lm):
            continue
        d = acc[t]["hi"] / acc[t]["hn"] - acc[t]["lo"] / acc[t]["ln"]
        nrm = np.linalg.norm(d)
        if nrm < 1e-8:
            continue
        dvec[t] = d / nrm
        calib_n[t] = (acc[t]["hn"], acc[t]["ln"])
        # 阴性对照：同一批检测细胞里随机抽同样数量的两组
        cells = list(all_map[t].keys())
        nh, nl = acc[t]["hn"], acc[t]["ln"]
        pick = rng.choice(len(cells), size=min(nh + nl, len(cells)), replace=False)
        picked = [cells[j] for j in pick]
        hi2 = picked[:nh]; lo2 = picked[nh:nh + nl]
        if len(hi2) and len(lo2):
            dn = (np.mean([emb_store[t][i] for i in hi2], axis=0)
                  - np.mean([emb_store[t][i] for i in lo2], axis=0))
            nn = np.linalg.norm(dn)
            dnull[t] = dn / nn if nn > 1e-8 else None
    log(f"方向向量构建: {len(dvec)} 个基因（含阴性对照 {sum(1 for v in dnull.values() if v is not None)}）")

    # ---------- Pass 2：pert 前向，投影 ----------
    log("Pass 2/2：扰动前向（VNN1 移至 rank1 + 截断语义与库内一致）...")
    pert_seqs, pert_len = [], []
    for i in order:
        seq = list(ids_all[i])
        if vnn1_tok in seq:
            seq.remove(vnn1_tok)
        seq.insert(1, vnn1_tok)
        if len(seq) > MAX_LEN:
            seq = seq[0:MAX_LEN - 1] + [seq[-1]]
        pert_seqs.append(seq)
        pert_len.append(len(seq))

    stats = {t: {"s": [], "s2": [], "sn": [], "sn2": [], "cos": [], "mag": [], "pair": []}
             for t in dvec}

    # 优化：避免逐细胞 seq.index（250 基因×3693 细胞太慢），预计算位置映射
    order_of = {i: oi for oi, i in enumerate(order)}
    log("预计算扰动序列位置映射 ...")
    pert_pos = [None] * n   # order_pos -> {token: pos}
    for oi, seq in enumerate(pert_seqs):
        pm = {}
        for pos, t in enumerate(seq):
            if t in dvec:
                pm[t] = pos
        pert_pos[oi] = pm

    def on_pert(embs, chunk):
        # 1) 收集本批每个细胞每个基因的 delta
        batch_deltas = []
        for bi, (i, _, _) in enumerate(chunk):
            pm = pert_pos[order_of[i]]
            dd = {}
            for t, p in pm.items():
                oe = emb_store[t].get(i)
                if oe is None:
                    continue
                dd[t] = embs[bi, p, :].astype(np.float32) - oe
            batch_deltas.append(dd)
        # 2) 逐细胞扣除公共位移分量（该细胞全部候选基因 delta 的均值，不含 VNN1 自身）
        for bi, (i, _, _) in enumerate(chunk):
            dd = batch_deltas[bi]
            if not dd:
                continue
            others = [v for t, v in dd.items() if t != vnn1_tok]
            gmean = np.mean(others, axis=0) if others else np.zeros(dim, dtype=np.float32)
            for t, delta in dd.items():
                d = dvec[t]
                dn = dnull.get(t)
                res = delta - gmean          # 去公共分量的残差
                s = float(res @ d)
                st = stats[t]
                st["s"].append(s); st["s2"].append(s * s)
                if dn is not None:
                    sn = float(res @ dn)
                    st["sn"].append(sn); st["sn2"].append(sn * sn)
                    st["pair"].append(s - sn)   # 配对差：秩校准轴 vs 随机轴
                st["cos"].append(s / max(float(np.linalg.norm(res)), 1e-12))
                st["mag"].append(float(np.linalg.norm(delta)))

    t0 = time.time()
    run_batches(lambda b0: seq_list[b0:b0 + BATCH], on_pert)
    log(f"Pass 2 完成，用时 {time.time()-t0:.0f}s")

    # ---------- 汇总 ----------
    from scipy.stats import norm

    rows = []
    for t, g in cand.items():
        st = stats.get(t)
        if not st or len(st["s"]) < 10:
            continue
        s = np.array(st["s"])
        cos = np.array(st["cos"]); mag = np.array(st["mag"])
        pair = np.array(st["pair"])
        m, sd = s.mean(), s.std(ddof=1)
        z = m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan
        # 配对差（秩轴 - 随机轴），去除校准轴构建伪影
        pm_, psd = pair.mean(), pair.std(ddof=1)
        pz = pm_ / (psd / np.sqrt(len(pair))) if len(pair) and psd > 0 else np.nan
        hn, ln = calib_n.get(t, (0, 0))
        rows.append({
            "gene": g, "token": t, "n_cells": len(s),
            "dir_score": m, "dir_sd": sd, "dir_z": z,
            "cos_dir_mean": cos.mean(),
            "delta_norm_mean": mag.mean(),
            "pair_mean": pm_, "pair_z": pz,
            "calib_hi_n": hn, "calib_lo_n": ln,
        })
    df = pd.DataFrame(rows)
    # BH-FDR（双侧）
    p = 2 * norm.sf(np.abs(df["dir_z"].to_numpy()))
    p = np.nan_to_num(p, nan=1.0)
    o = np.argsort(p); nn_ = len(p)
    ranked = p[o] * nn_ / (np.arange(nn_) + 1)
    q = np.empty(nn_); q[o] = np.minimum.accumulate(ranked[::-1])[::-1]
    df["dir_p"] = p; df["dir_q"] = q
    df["direction"] = np.where((df["dir_z"] > 0) & (df["dir_q"] < 0.05), "UP",
                        np.where((df["dir_z"] < 0) & (df["dir_q"] < 0.05), "DOWN", "NS"))
    df = df.sort_values("dir_z", ascending=False)
    out_csv = OUT_DIR / "vnn1_oe_direction.csv"
    df.to_csv(out_csv, index=False)
    n_up = (df["direction"] == "UP").sum(); n_dn = (df["direction"] == "DOWN").sum()
    log(f"\n方向表 -> {out_csv}  ({len(df)} 行, UP {n_up} / DOWN {n_dn} / NS {len(df)-n_up-n_dn})")

    pd.set_option("display.width", 200)
    show = ["gene", "n_cells", "dir_score", "dir_z", "dir_q", "cos_dir_mean",
            "delta_norm_mean", "pair_z", "direction"]
    log("\n【上调最显著的 15 个】")
    log(df.head(15)[show].round(4).to_string(index=False))
    log("\n【下调最显著的 15 个】")
    log(df.tail(15)[show].round(4).to_string(index=False))
    v = df[df["gene"] == "VNN1"]
    if len(v):
        log("\n【阳性对照 VNN1】")
        log(v[show].round(4).to_string(index=False))
    log("\n【关键基因方向】")
    key_rows = df[df["gene"].isin(KEY_GENES)]
    log(key_rows[show].round(4).to_string(index=False) if len(key_rows) else "  无")

    # 方向向量保存
    np.savez_compressed(
        OUT_DIR / "vnn1_oe_direction_vectors.npz",
        tokens=np.array(list(dvec.keys())),
        dvec=np.stack([dvec[t] for t in dvec]),
        dnull=np.stack([dnull[t] for t in dnull if dnull[t] is not None])
        if any(v is not None for v in dnull.values()) else np.zeros((0, dim)),
    )
    log(f"\n总用时 {(time.time()-t00)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
