"""
GSE189175 肿瘤组织恶性细胞（tumor-only malignant）VNN1 过表达流水线。
与 scripts/18 同参数，仅数据源与输出目录不同（独立目录避免实验间污染）。
"""

import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))

VNN1 = "ENSG00000112299"
MODEL = ROOT / "models" / "V2-104M"
DATA = ROOT / "data"
SRC_DIR = DATA / "nashhcc_counts" / "tumor_mal"
DS_DIR = DATA / "tokenized"
DS_NAME = "nashhcc_tumor_mal"
OUT = ROOT / "outputs" / "nashhcc_tumor_isp"
RAW_DIR = OUT / "raw"
GENE_DIR = OUT / "gene"
CELL_DIR = OUT / "cell"
STATS_DIR = ROOT / "outputs" / "nashhcc_tumor_isp_stats"
BATCH = 8
ATTRS = {"cell_type": "cell_type", "malignant": "malignant", "SampleID": "SampleID",
         "Tissue": "Tissue", "Condition": "Condition", "Patient": "Patient"}


def log(msg=""):
    print(msg, flush=True)


def banner(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def step1_tokenize():
    from geneformer import TranscriptomeTokenizer

    out_ds = DS_DIR / f"{DS_NAME}.dataset"
    if out_ds.exists():
        log(f"[skip] {out_ds} 已存在")
        return
    banner(f"步骤 1：tokenize {DS_NAME}")
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=ATTRS, nproc=8, model_version="V2",
        use_h5ad_index=True, special_token=True,
    )
    t0 = time.time()
    tk.tokenize_data(str(SRC_DIR), str(DS_DIR), DS_NAME, file_format="h5ad")
    log(f"tokenize 完成，用时 {time.time()-t0:.0f}s -> {out_ds}")


def step2_check():
    import numpy as np
    from datasets import load_from_disk

    banner("步骤 2：tokenized 数据体检")
    tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id = tok["<cls>"], tok["<eos>"]
    vnn1_tok = tok[VNN1]
    log(f"<cls>={cls_id}  <eos>={eos_id}  VNN1 token={vnn1_tok}")

    ds = load_from_disk(str(DS_DIR / f"{DS_NAME}.dataset"))
    ids = ds["input_ids"]
    lens = [len(x) for x in ids]
    n_cls = sum(1 for x in ids if x[0] == cls_id)
    n_eos = sum(1 for x in ids if x[-1] == eos_id)
    n_vnn1 = sum(1 for x in ids if vnn1_tok in x)
    log(f"细胞 {len(ds)}  长度 中位={float(np.median(lens)):.0f} 最大={max(lens)}  "
        f"<cls>开头 {n_cls}  <eos>结尾 {n_eos}  含VNN1 {n_vnn1} ({n_vnn1/len(ds):.1%})")
    assert n_cls == len(ds) and n_eos == len(ds), "特殊 token 缺失！"


def step3_perturb():
    from geneformer import InSilicoPerturber
    from geneformer_safe import vram_report

    banner("步骤 3：overexpress VNN1（tumor malignant 细胞, fp32, cls_and_gene）")
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
        isp.perturb_data(str(MODEL), str(DS_DIR / f"{DS_NAME}.dataset"),
                         str(RAW_DIR), "vnn1_oe")
        log(f"扰动完成，用时 {time.time()-t0:.0f}s")


def step4_stats():
    from geneformer import InSilicoPerturberStats

    banner("步骤 4：统计下游基因位移与细胞位移")
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
