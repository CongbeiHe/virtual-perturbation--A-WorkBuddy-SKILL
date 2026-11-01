"""
肝 HCC 数据集 VNN1 过表达扰动主流水线（在独立终端窗口中运行，供实时观察）。

步骤：
  1. tokenize 两份数据（normal + tumor）
  2. tokenized 数据体检（cls/eos、VNN1 token、序列长度）
  3. 对 tumor 全部 1207 细胞做 overexpress VNN1（cls_and_gene，fp32）
  4. 拆分中间 pickle，统计下游基因位移 + 细胞级位移
"""

import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))

VNN1 = "ENSG00000112299"
MODEL = ROOT / "models" / "V2-104M"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "liver_isp"
RAW_DIR = OUT / "tumor"            # perturb_data 的原始 pickle 输出
GENE_DIR = OUT / "tumor_gene"      # 基因级 pickle（下游基因分析用）
CELL_DIR = OUT / "tumor_cell"      # 细胞级 pickle（细胞位移分析用）
STATS_DIR = ROOT / "outputs" / "liver_isp_stats"
BATCH = 8


def log(msg=""):
    print(msg, flush=True)


def banner(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def step1_tokenize():
    from geneformer import TranscriptomeTokenizer

    attrs = {"cell_type": "cell_type", "author_cell_type": "author_cell_type",
             "leiden_1": "leiden_1", "donor_id": "donor_id", "disease": "disease"}
    for tag in ["normal", "tumor"]:
        out_ds = DATA / "tokenized" / f"liver_{tag}.dataset"
        if out_ds.exists():
            log(f"[skip] {out_ds} 已存在")
            continue
        banner(f"步骤 1a：tokenize liver_{tag}")
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=attrs, nproc=8, model_version="V2",
            use_h5ad_index=True, special_token=True,
        )
        t0 = time.time()
        tk.tokenize_data(str(DATA / "liver_counts" / tag), str(DATA / "tokenized"),
                         f"liver_{tag}", file_format="h5ad")
        log(f"tokenize 完成，用时 {time.time()-t0:.0f}s -> {out_ds}")


def step2_check():
    from datasets import load_from_disk

    banner("步骤 2：tokenized 数据体检")
    tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id = tok["<cls>"], tok["<eos>"]
    vnn1_tok = tok[VNN1]
    log(f"<cls>={cls_id}  <eos>={eos_id}  VNN1 token={vnn1_tok}")

    for tag in ["normal", "tumor"]:
        ds = load_from_disk(str(DATA / "tokenized" / f"liver_{tag}.dataset"))
        ids = ds["input_ids"]
        lens = [len(x) for x in ids]
        n_cls = sum(1 for x in ids if x[0] == cls_id)
        n_eos = sum(1 for x in ids if x[-1] == eos_id)
        n_vnn1 = sum(1 for x in ids if vnn1_tok in x)
        log(f"[{tag}] 细胞 {len(ds)}  长度 中位={int(np_median(lens))} 最大={max(lens)}  "
            f"<cls>开头 {n_cls}  <eos>结尾 {n_eos}  含VNN1 {n_vnn1}")
        assert n_cls == len(ds) and n_eos == len(ds), "特殊 token 缺失！"


def np_median(x):
    import numpy as np
    return float(np.median(x))


def step3_perturb():
    from geneformer import InSilicoPerturber
    from geneformer_safe import vram_report

    banner("步骤 3：overexpress VNN1（全部肿瘤细胞, fp32, cls_and_gene）")
    info = vram_report()
    log(f"设备: {info['device']}  总显存 {info['total_gb']}G  batch={BATCH}")

    for d in [RAW_DIR, GENE_DIR, CELL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if any(RAW_DIR.glob("*_raw.pickle")):
        log("[skip] 中间 pickle 已存在，跳过扰动计算")
    else:
        isp = InSilicoPerturber(
            perturb_type="overexpress",
            perturb_rank_shift=None,
            genes_to_perturb=[VNN1],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls_and_gene",
            cell_emb_style="mean_pool",
            filter_data=None,
            cell_states_to_model=None,
            state_embs_dict=None,
            max_ncells=None,
            cell_inds_to_perturb="all",
            emb_layer=0,
            forward_batch_size=BATCH,
            nproc=8,
            model_version="V2",
        )
        t0 = time.time()
        isp.perturb_data(str(MODEL), str(DATA / "tokenized" / "liver_tumor.dataset"),
                         str(RAW_DIR), "vnn1_oe")
        log(f"扰动完成，用时 {time.time()-t0:.0f}s")


def step4_stats():
    from geneformer import InSilicoPerturberStats

    banner("步骤 4：统计下游基因位移与细胞位移")
    # 拆分 cell / gene pickle，避免 read_dict(gene) 把 cell_emb 混进来
    for f in RAW_DIR.glob("*_raw.pickle"):
        dst = GENE_DIR if "_gene_embs_dict_" in f.name else CELL_DIR
        shutil.copy2(f, dst / f.name)
    log(f"gene pickle: {len(list(GENE_DIR.glob('*_raw.pickle')))} 个, "
        f"cell pickle: {len(list(CELL_DIR.glob('*_raw.pickle')))} 个")

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    st = InSilicoPerturberStats(
        mode="aggregate_gene_shifts", genes_perturbed=[VNN1], combos=0,
        anchor_gene=None, cell_states_to_model=None,
        pickle_suffix="_raw.pickle", model_version="V2",
    )
    st.get_stats(str(GENE_DIR), None, str(STATS_DIR), "vnn1_oe_gene_shifts")
    log(f"下游基因表 -> {STATS_DIR / 'vnn1_oe_gene_shifts.csv'}")

    st2 = InSilicoPerturberStats(
        mode="aggregate_data", genes_perturbed=[VNN1], combos=0,
        anchor_gene=None, cell_states_to_model=None,
        pickle_suffix="_raw.pickle", model_version="V2",
    )
    st2.get_stats(str(CELL_DIR), None, str(STATS_DIR), "vnn1_oe_cell_shift")
    log(f"细胞位移表 -> {STATS_DIR / 'vnn1_oe_cell_shift.csv'}")


def step5_preview():
    import pandas as pd

    banner("步骤 5：结果预览")
    pd.set_option("display.width", 200)
    g = pd.read_csv(STATS_DIR / "vnn1_oe_gene_shifts.csv")
    log(f"下游基因表: {g.shape[0]} 行")
    cols = ["Perturbed", "Affected_gene_name", "Affected_Ensembl_ID",
            "Cosine_sim_mean", "Cosine_sim_stdev", "N_Detections"]
    cols = [c for c in cols if c in g.columns]
    log("位移最大的 20 个基因（Cosine_sim_mean 越小 = 表征变化越大）:")
    log(g.nsmallest(20, "Cosine_sim_mean")[cols].to_string(index=False))

    c = pd.read_csv(STATS_DIR / "vnn1_oe_cell_shift.csv")
    log(f"\n细胞位移表: {c.shape[0]} 行")
    log(c.head(10).to_string(index=False))


if __name__ == "__main__":
    t00 = time.time()
    step1_tokenize()
    step2_check()
    step3_perturb()
    step4_stats()
    step5_preview()
    banner(f"全部完成，总用时 {(time.time()-t00)/60:.1f} 分钟")
