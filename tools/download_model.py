r"""
Geneformer 模型权重下载脚本（国内镜像版）

用法:
    python tools/download_model.py                    # 默认下载 Geneformer-V2-104M
    python tools/download_model.py V1-10M             # 下载 V1-10M (41MB, 快速验证用)
    python tools/download_model.py V2-316M            # 下载 V2-316M (1.2GB)
    python tools/download_model.py V2-104M_CLcancer   # 下载癌症领域持续学习版
    python tools/download_model.py --list             # 列出可下载的模型

说明:
    官方权重托管在 HuggingFace，国内直连 huggingface.co 不通，
    脚本默认走 hf-mirror.com 镜像。如需切换，设置环境变量 HF_ENDPOINT。
"""

import argparse
import os
import ssl
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
REPO = "ctheodoris/Geneformer"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# 模型别名 -> 仓库内子目录；"." 表示仓库根目录即该模型
MODELS = {
    "V1-10M": "Geneformer-V1-10M",
    "V2-104M": "Geneformer-V2-104M",
    "V2-104M_CLcancer": "Geneformer-V2-104M_CLcancer",
    "V2-316M": "Geneformer-V2-316M",
    "V2-316M-root": ".",
}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
}


def _api(path):
    import json

    url = f"{HF_ENDPOINT}/api/models/{REPO}/tree/main"
    if path:
        url += "/" + urllib.parse.quote(path)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
        return json.loads(r.read().decode())


def list_models():
    print("可下载的模型（来自 %s）:\n" % HF_ENDPOINT)
    for alias, sub in MODELS.items():
        try:
            entries = _api("" if sub == "." else sub)
            size = sum(e.get("size", 0) for e in entries if e["type"] == "file")
            print(f"  {alias:<20} {size / 1048576:8.1f} MB   {sub}/")
        except Exception as exc:  # pragma: no cover
            print(f"  {alias:<20} (查询失败: {exc})")


def download(alias):
    if alias not in MODELS:
        print(f"未知模型 '{alias}'，可选: {', '.join(MODELS)}")
        sys.exit(1)

    sub = MODELS[alias]
    dest = os.path.join(MODEL_DIR, alias)
    os.makedirs(dest, exist_ok=True)

    entries = [e for e in _api("" if sub == "." else sub) if e["type"] == "file"]
    entries = [e for e in entries if e["path"].rsplit("/", 1)[-1] != "training_args.bin"]
    total = sum(e["size"] for e in entries)
    print(f"[{alias}] 共 {len(entries)} 个文件 / {total / 1048576:.1f} MB -> {dest}")

    def fetch(entry):
        name = os.path.basename(entry["path"])
        out = os.path.join(dest, name)
        if os.path.exists(out) and os.path.getsize(out) == entry["size"]:
            return f"  = 已存在  {name}"
        url = f"{HF_ENDPOINT}/{REPO}/resolve/main/{urllib.parse.quote(entry['path'])}"
        req = urllib.request.Request(url, headers=_HEADERS)
        tmp = out + ".part"
        with urllib.request.urlopen(req, timeout=300, context=_CTX) as r, open(tmp, "wb") as w:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                w.write(chunk)
        os.replace(tmp, out)
        return f"  v 完成    {name} ({entry['size'] / 1048576:.1f} MB)"

    with ThreadPoolExecutor(max_workers=3) as ex:
        for msg in ex.map(fetch, entries):
            print(msg, flush=True)

    print(f"\n[{alias}] 下载完成: {dest}")
    print("加载示例:")
    print(f'  AutoModelForMaskedLM.from_pretrained(r"{dest}")')


def main():
    ap = argparse.ArgumentParser(description="Geneformer 权重下载器")
    ap.add_argument("model", nargs="?", default="V2-104M", help="模型别名，默认 V2-104M")
    ap.add_argument("--list", action="store_true", help="列出所有可用模型")
    args = ap.parse_args()
    if args.list:
        list_models()
    else:
        download(args.model)


if __name__ == "__main__":
    main()
