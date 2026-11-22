"""Llama-3.1-8B-Instruct, Hugging Face transformers, CUDA bf16, for the D71 stage-3 run (RunPod).

Same call shape as ``sensos_inference.Llama31MLXProvider.generate_chat`` and the same
``InferenceResult``. Generation is delegated to the *frozen* MSD0001 backend
(``msd.backends.HFBackend``, tag ``msd0001-v1.1-frozen``, loaded unchanged from ``MSD_SRC``): HF
``generate``, ``do_sample``, temperature 0.7, top_p 0.95 (``msd.config``), layer 32 hidden states.
This module is a thin wrapper: it does not change the sampler, so only the *seed value* is shared with
stage 2 (the random streams differ: torch multinomial here, NumPy PCG64 on MLX; v0.2-A 3.2.1).

Not runnable on the Mac (no GPU). The backend is injectable, which is how the tests run.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from sensos_inference.inference import (
    EOS_MISSING,
    GENERATE_FAILED,
    LENGTH_LIMIT,
    LOAD_FAILED,
    InferenceResult,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
LAYER = 32
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.95, 50  # TOP_K is the HF default (stage 2 used the same value)


def deterministic_env() -> None:
    """Must run before CUDA is initialised (v0.2-B 1). Environment part only; torch part below."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")


#: Tomo's decision (a), 2026-10-15, taken AFTER D71-B0 failed with ``warn_only=False`` (cumsum has no
#: deterministic CUDA kernel). v0.2-B 1 had ``warn_only=False``. Warnings are counted, never dropped.
WARN_ONLY = True
_OP_RE = re.compile(r"(\w+) does not have a deterministic implementation")


def deterministic_torch(torch: Any, *, warn_only: bool = WARN_ONLY) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "use_deterministic_algorithms": True, "warn_only": warn_only, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def _frozen_backend(model_id: str, revision: str) -> Any:  # pragma: no cover - needs GPU
    src = os.environ.get("MSD_SRC")
    if not src:
        raise RuntimeError("MSD_SRC (path to the msd0001-v1.1-frozen 'src' directory) is not set")
    if src not in sys.path:
        sys.path.insert(0, src)
    from msd.backends import HFBackend  # type: ignore[import-not-found]

    return HFBackend(model_id=model_id, revision=revision, layers=(LAYER,), device="cuda", dtype="bfloat16")


def summarize_warnings(caught: list[Any]) -> dict[str, Any]:
    ops: dict[str, int] = {}
    msgs: list[str] = []
    for w in caught:
        text = str(w.message)
        m = _OP_RE.search(text)
        if m:
            ops[m.group(1)] = ops.get(m.group(1), 0) + 1
        if len(msgs) < 3:
            msgs.append(text[:200])
    return {"count": len(caught), "ops": ops, "messages": msgs}


class Llama31HFProvider:
    name = "llama31-hf-bf16"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        revision: str | None = REVISION,
        backend_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._factory = backend_factory or _frozen_backend
        self._backend: Any = None
        self._error: str | None = None
        self._load_seconds: float | None = None
        self.last_hidden: np.ndarray | None = None  # [T, 1, 4096] float32 (layer 32), see v0.2-B 2
        self.last_logprob: np.ndarray | None = None
        self.last_entropy: np.ndarray | None = None
        self.last_warnings: dict[str, Any] = {"count": 0, "ops": {}, "messages": []}

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        if self._error is not None:
            raise RuntimeError(self._error)
        t0 = time.perf_counter()
        try:
            self._backend = self._factory(self.model_id, self.revision or "")
        except Exception as exc:
            self._error = f"load failed: {exc}"
            raise RuntimeError(self._error) from exc
        self._load_seconds = time.perf_counter() - t0

    def _eos_ids(self) -> set[int]:
        b = self._backend
        ids: set[int] = set()
        explicit = getattr(b, "eos_ids", None)
        if callable(explicit):
            return {int(i) for i in explicit()}
        gen = getattr(getattr(b, "model", None), "generation_config", None)
        raw = getattr(gen, "eos_token_id", None)
        for v in (raw if isinstance(raw, (list, tuple)) else [raw]):
            if v is not None:
                ids.add(int(v))
        tok = getattr(getattr(b, "tok", None), "eos_token_id", None)
        if tok is not None:
            ids.add(int(tok))
        return ids

    def chat_template_sha256(self) -> str | None:
        template = getattr(getattr(self._backend, "tok", None), "chat_template", None)
        return hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None

    def describe(self) -> dict[str, Any]:
        d = getattr(self._backend, "describe", None)
        return dict(d()) if callable(d) else {}

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 768,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
        top_p: float = TOP_P,
        seed: int = 0,
    ) -> InferenceResult:
        base: dict[str, Any] = {
            "model": self.model_id, "model_ref": self.model_id, "revision": self.revision,
            "prompt_chars": sum(len(m.get("content", "")) for m in messages), "max_tokens": max_tokens,
            "sampler": {"impl": "hf_generate(torch multinomial)", "temperature": TEMPERATURE, "top_p": TOP_P,
                        "top_k": TOP_K, "do_sample": True, "min_new_tokens": 0, "seed": seed},
        }
        if (temperature, top_p, top_k) != (TEMPERATURE, TOP_P, TOP_K):
            return InferenceResult(
                ok=False, text="", failure_kind=GENERATE_FAILED, **base,
                error="the frozen MSD backend fixes temperature/top_p; refusing a different request")
        try:
            self._ensure_loaded()
        except Exception as exc:
            return InferenceResult(ok=False, text="", error=str(exc), failure_kind=LOAD_FAILED, **base)
        eos = self._eos_ids()
        base.update(device="cuda", load_seconds=self._load_seconds, chat_template_sha256=self.chat_template_sha256())
        if not eos:
            return InferenceResult(ok=False, text="", error="no EOS token ids", failure_kind=EOS_MISSING, **base)
        try:
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")  # count every occurrence
                out = self._backend.generate(messages, seed=seed, max_new_tokens=max_tokens, min_new_tokens=0)
            seconds = time.perf_counter() - t0
            self.last_warnings = summarize_warnings(caught)
        except Exception as exc:  # a generation failure; the runner stops on it
            return InferenceResult(ok=False, text="", error=str(exc), failure_kind=GENERATE_FAILED, **base)
        ids = [int(i) for i in np.asarray(out.ids).tolist()]
        finish: str | None = None
        if ids and ids[-1] in eos:
            ids, finish = ids[:-1], "eos"
        n_total = len(ids) + (1 if finish == "eos" else 0)
        if finish is None and n_total >= max_tokens:
            finish = "length"
        keep = len(ids)
        hidden = np.asarray(out.hidden, dtype=np.float32)
        if hidden.ndim == 2:  # [T, d] -> [T, 1, d]
            hidden = hidden[:, None, :]
        self.last_hidden = hidden  # includes the EOS step if any (one vector per generated token)
        self.last_logprob = np.asarray(out.logprob, dtype=np.float32)
        self.last_entropy = np.asarray(out.entropy, dtype=np.float32)
        kind: str | None = None
        if finish == "length":
            kind = LENGTH_LIMIT
        elif finish is None:
            kind = EOS_MISSING
        return InferenceResult(
            ok=finish == "eos", text=str(out.text), token_ids=ids,
            prompt_tokens=None, completion_tokens=keep, generate_seconds=seconds, finish_reason=finish,
            failure_kind=kind, error=None if kind is None else f"generation ended without EOS ({kind})", **base,
        )
