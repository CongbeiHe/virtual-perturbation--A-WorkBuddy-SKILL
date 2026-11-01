"""
肝 HCC 数据集 FGB 过表达扰动主流水线（与 VNN1 流水线 11 同参数，独立输出目录）。

FGB (fibrinogen beta chain) = ENSG00000171564, token 12844

步骤：
  1. tokenized 数据体检（复用 11 已生成的 liver_normal/tumor.dataset）
  2. 对 tumor 全部 1207 细胞做 overexpress FGB（cls_and_gene，fp32）
  3. 拆分中间 pickle，统计下游基因位移 + 细胞级位移
"""

import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))

FGB = "ENSG00000171564"
MODEL = ROOT / "models" / "V2-104M"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "liver_isp_fgb"
RAW_DIR = OUT / "tumor"
GENE_DIR = OUT / "tumor_gene"
CELL_DIR = OUT / "tumor_cell"
STATS_DIR = ROOT / "outputs" / "liver_isp_fgb_stats"
BATCH = 8


def log(msg=""):
    print(msg, flush=True)


def banner(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def step1_check():
    import numpy as np
    from datasets import load_from_disk

    banner("步骤 1：tokenized 数据体检（FGB）")
    tok = pickle.load(open(ROOT / "code" / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id = tok["<cls>"], tok["<eos>"]
    fgb_tok = tok[FGB]
    log(f"<cls>={cls_id}  <eos>={eos_id}  FGB token={fgb_tok}")

    for tag in ["normal", "tumor"]:
        ds = load_from_disk(str(DATA / "tokenized" / f"liver_{tag}.dataset"))
        ids = ds["input_ids"]
        lens = [len(x) for x in ids]
        n_cls = sum(1 for x in ids if x[0] == cls_id)
        n_eos = sum(1 for x in ids if x[-1] == eos_id)
        n_fgb = sum(1 for x in ids if fgb_tok in x)
        log(f"[{tag}] 细胞 {len(ds)}  长度 中位={float(np.median(lens)):.0f} 最大={max(lens)}  "
            f"<cls>开头 {n_cls}  <eos>结尾 {n_eos}  含FGB {n_fgb}")
        assert n_cls == len(ds) and n_eos == len(ds), "特殊 token 缺失！"

    # 原始计数口径的表达情况
    import anndata as ad
    a = ad.read_h5ad(ROOT / "data" / "liver_counts" / "tumor" / "liver_tumor.h5ad")
    X = a.X.tocsc()
    j = list(map(str, a.var_names)).index(FGB)
    col = np.asarray(X[:, j].todense()).ravel()
    log(f"[tumor raw counts] FGB 表达细胞数 {(col > 0).sum()} / {a.n_obs}  "
        f"max count {int(col.max())}  总 counts {int(col.sum())}")


def step2_perturb():
    from geneformer import InSilicoPerturber
    from geneformer_safe import vram_report

    banner("步骤 2：overexpress FGB（全部肿瘤细胞, fp32, cls_and_gene）")
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
            genes_to_perturb=[FGB],
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
                         str(RAW_DIR), "fgb_oe")
        log(f"扰动完成，用时 {time.time()-t0:.0f}s")


def step3_stats():
    from geneformer import InSilicoPerturberStats

    banner("步骤 3：统计下游基因位移与细胞位移")
    for f in RAW_DIR.glob("*_raw.pickle"):
        dst = GENE_DIR if "_gene_embs_dict_" in f.name else CELL_DIR
        shutil.copy2(f, dst / f.name)
    log(f"gene pickle: {len(list(GENE_DIR.glob('*_raw.pickle')))} 个, "
        f"cell pickle: {len(list(CELL_DIR.glob('*_raw.pickle')))} 个")

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    st = InSilicoPerturberStats(
        mode="aggregate_gene_shifts", genes_perturbed=[FGB], combos=0,
        anchor_gene=None, cell_states_to_model=None,
        pickle_suffix="_raw.pickle", model_version="V2",
    )
    st.get_stats(str(GENE_DIR), None, str(STATS_DIR), "fgb_oe_gene_shifts")
    log(f"下游基因表 -> {STATS_DIR / 'fgb_oe_gene_shifts.csv'}")

    st2 = InSilicoPerturberStats(
        mode="aggregate_data", genes_perturbed=[FGB], combos=0,
        anchor_gene=None, cell_states_to_model=None,
        pickle_suffix="_raw.pickle", model_version="V2",
    )
    st2.get_stats(str(CELL_DIR), None, str(STATS_DIR), "fgb_oe_cell_shift")
    log(f"细胞位移表 -> {STATS_DIR / 'fgb_oe_cell_shift.csv'}")


def step4_preview():
    import pandas as pd

    banner("步骤 4：结果预览")
    pd.set_option("display.width", 200)
    g = pd.read_csv(STATS_DIR / "fgb_oe_gene_shifts.csv")
    log(f"下游基因表: {g.shape[0]} 行")
    cols = ["Perturbed", "Affected_gene_name", "Affected_Ensembl_ID",
            "Cosine_sim_mean", "Cosine_sim_stdev", "N_Detections"]
    cols = [c for c in cols if c in g.columns]
    log("位移最大的 20 个基因（Cosine_sim_mean 越小 = 表征变化越大）:")
    log(g.nsmallest(20, "Cosine_sim_mean")[cols].to_string(index=False))

    c = pd.read_csv(STATS_DIR / "fgb_oe_cell_shift.csv")
    log(f"\n细胞位移表: {c.shape[0]} 行")
    log(c.head(10).to_string(index=False))


if __name__ == "__main__":
    t00 = time.time()
    step1_check()
    step2_perturb()
    step3_stats()
    step4_preview()
    banner(f"全部完成，总用时 {(time.time()-t00)/60:.1f} 分钟")
