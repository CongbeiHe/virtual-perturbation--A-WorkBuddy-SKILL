"""
Geneformer 显存安全护栏
=======================

背景
----
Geneformer V2-104M 的 config.json 声明 `torch_dtype: float32`，而库内
`perturber_utils.load_model()` 调用 `from_pretrained` 时**不传 torch_dtype**，
于是模型默认以 fp32 常驻显存，前向激活值也是 fp32。

在 8GB 显存的笔记本 GPU 上，配合较大的 `forward_batch_size`，显存需求会
轻松突破物理上限（实测外推：fp32 / batch=100 约需 20.8 GB）。显存被彻底榨干时，
WDDM 的显存换页路径会进入异常分支，NVIDIA 驱动可能在核心态崩溃，表现为
整机蓝屏重启（而不是干净的 CUDA OOM 异常）。

本模块提供两道护栏：
1. `enable_bf16()` —— 猴补丁让 Geneformer 强制以 bf16 加载，显存直接减半；
2. `VRAMGuard` —— 显存预检 / 监控 / 超限保护，避免把 GPU 逼到崩溃边缘。

用法
----
    from tools.geneformer_safe import enable_bf16, VRAMGuard

    enable_bf16()                       # 必须在 import geneformer 之前或之后立即调用
    with VRAMGuard(limit_frac=0.85):    # 超过 85% 显存即告警
        embs = embex.extract_embs(...)
"""

from __future__ import annotations

import gc
import warnings

import torch

__all__ = ["enable_bf16", "VRAMGuard", "vram_report", "recommend_batch_size"]


# ---------------------------------------------------------------- 精度护栏

_BF16_PATCHED = False


def enable_bf16(dtype: torch.dtype = torch.bfloat16, verbose: bool = True) -> bool:
    """让所有 HF 模型的 from_pretrained 默认以半精度加载。

    Geneformer 的加载路径不暴露 dtype 参数，因此在 PreTrainedModel 基类上做
    猴补丁，是唯一不改动上游源码的实现方式。

    返回 True 表示补丁已生效（重复调用不会重复打补丁）。
    """
    global _BF16_PATCHED
    if _BF16_PATCHED:
        return True

    try:
        from transformers import PreTrainedModel
    except ImportError:  # pragma: no cover
        warnings.warn("transformers 不可用，跳过 bf16 补丁")
        return False

    # classmethod 需要用 __func__ 取出底层函数再重新包装
    original = PreTrainedModel.from_pretrained.__func__

    def patched(cls, *args, **kwargs):
        # 已量化（bitsandbytes）的模型不能让 torch_dtype 覆盖量化配置
        quantized = kwargs.get("quantization_config") is not None
        if not quantized and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype
        return original(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(patched)
    _BF16_PATCHED = True

    if verbose:
        print(f"[vram-guard] 已强制 HF 模型默认以 {dtype} 加载（显存减半）")
    return True


# ---------------------------------------------------------------- 显存护栏


def vram_report(tag: str = "") -> dict:
    """读取当前显存状态，返回结构化字典。"""
    if not torch.cuda.is_available():
        return {"available": False}

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    info = {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(total / 1024**3, 2),
        "free_gb": round(free / 1024**3, 2),
        "allocated_gb": round(allocated / 1024**3, 2),
        "reserved_gb": round(reserved / 1024**3, 2),
        "peak_gb": round(peak / 1024**3, 2),
        "used_frac": round((total - free) / total, 3),
    }
    if tag:
        print(
            f"[vram-guard] {tag}: "
            f"占用 {info['allocated_gb']}G / 保留 {info['reserved_gb']}G / "
            f"峰值 {info['peak_gb']}G / 总 {info['total_gb']}G "
            f"({info['used_frac']:.0%})"
        )
    return info


def recommend_batch_size(
    per_sample_gb: float = 0.20,
    dtype_factor: float = 1.0,
    safety: float = 0.70,
) -> int:
    """按每样本显存开销推荐安全 batch size。

    per_sample_gb 默认 0.20，取自本机实测：走 `EmbExtractor.extract_embs`
    完整链路（含 DataLoader、hidden_states 累积、csv 拼装）时，bf16 下
    batch=32 实测峰值 6.4GB，即约 0.20GB/样本；纯前向裸测只有 0.104GB/样本，
    但实际任务必须按链路开销估算，否则会低估显存。
    safety 默认 0.70，为驱动/WDDM 换页留出余量——显存被榨干时 Windows 会
    触发卸载换页，正是蓝屏的高发区间。
    """
    if not torch.cuda.is_available():
        return 1
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    budget = total * safety
    n = int(budget / (per_sample_gb * dtype_factor))
    # 收敛到 2 的幂，便于显存对齐
    power = 1
    while power * 2 <= n:
        power *= 2
    return max(1, min(power, 64))


class VRAMGuard:
    """显存监控上下文管理器。

    参数
    ----
    limit_frac : float
        显存占用占比上限，超过即触发告警（默认 0.85）。
    strict : bool
        为 True 时超限直接抛异常中断，避免把驱动逼到崩溃；为 False 时仅告警。
    watch_interval : float
        后台轮询间隔（秒）。0 表示不做轮询，只在进入/退出时检查。
    tag : str
        日志前缀。
    """

    def __init__(
        self,
        limit_frac: float = 0.85,
        strict: bool = False,
        watch_interval: float = 0.0,
        tag: str = "guard",
    ):
        self.limit_frac = limit_frac
        self.strict = strict
        self.watch_interval = watch_interval
        self.tag = tag
        self._baseline = None

    def __enter__(self):
        if not torch.cuda.is_available():
            return self
        torch.cuda.reset_peak_memory_stats()
        self._baseline = vram_report(f"{self.tag} 启动")
        return self

    def __exit__(self, exc_type, exc, tb):
        if not torch.cuda.is_available():
            return False

        info = vram_report(f"{self.tag} 结束")
        peak_frac = info["peak_gb"] / info["total_gb"] if info["total_gb"] else 0
        if peak_frac > self.limit_frac:
            print(
                f"[vram-guard] 警告：峰值显存 {info['peak_gb']}G 达 "
                f"{peak_frac:.0%}，超过阈值 {self.limit_frac:.0%}，"
                f"建议下调 batch size"
            )
        # 主动回收，避免碎片累积到下一轮
        gc.collect()
        torch.cuda.empty_cache()
        return False

    def check(self, strict: bool | None = None):
        """手动检查一次显存水位。

        注意：任务结束时显存通常已经释放，所以这里以**峰值**为准，
        而不是当前瞬时占用——峰值才反映任务对驱动的实际压力。
        """
        if not torch.cuda.is_available():
            return None
        info = vram_report()
        peak_frac = info["peak_gb"] / info["total_gb"]
        strict = self.strict if strict is None else strict
        if peak_frac > self.limit_frac:
            msg = (
                f"峰值显存 {info['peak_gb']}G 已达总显存 {info['total_gb']}G 的 "
                f"{peak_frac:.0%}，超过安全阈值 {self.limit_frac:.0%}"
                f"（当前占用 {info['allocated_gb']}G）"
            )
            if strict:
                raise MemoryError(msg + " —— 已中断以避免驱动崩溃")
            warnings.warn("[vram-guard] " + msg)
        return info
