"""Honour sampling parameters DFlash's speculative path silently drops.

``serve.py`` parses ``repetition_penalty``, ``presence_penalty``,
``frequency_penalty``, ``xtc_probability``, ``xtc_threshold``, ``logit_bias``,
``logprobs`` and ``top_logprobs`` off the request body, but only a subset ever
reaches ``stream_dflash_generate_impl``. The remainder are accepted and quietly
ignored -- not rejected, not applied -- so a client asking for them gets output
that silently disregards the request.

As of dflash-mlx 0.1.10+omlx.7 the speculative loop handles ``temperature``,
``top_p``, ``top_k``, ``min_p`` and ``repetition_penalty`` itself, so those keep
the fast path. Still dropped: the presence and frequency penalties, XTC,
``logit_bias`` and logprobs.

Rather than reimplement those inside the speculative loop, route such requests
to plain autoregressive decode on the target, which is mlx-lm's own server path
and does honour them. That is exactly the mechanism the upstream
``--fastpath-max-tokens`` option already uses: drop the draft model for the
duration of the request and delegate to the parent handler.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_GUARD_ENV = "LOCAL_LLM_DFLASH_EXACT_SAMPLING"

# mlx-lm's GenerationArguments nests these: penalties and logit_bias live on
# ``args.logits`` (LogitsProcessorArguments) and XTC on ``args.sampling``
# (SamplingArguments), while logprobs stays top level. Read the nested holder
# first and fall back to a flat attribute, so this keeps working whichever shape
# a given mlx-lm release hands over. repetition_penalty is deliberately absent:
# the runtime now applies it itself (engine/sampling.py::apply_repetition_penalty),
# so diverting those requests would throw away speculation for nothing.
_NEUTRAL_PENALTIES = (
    ("logits", "presence_penalty", (0.0,)),
    ("logits", "frequency_penalty", (0.0,)),
    ("sampling", "xtc_probability", (0.0,)),
)


def _resolve(args: Any, holder: str, field: str) -> Any:
    """Read ``args.<holder>.<field>``, falling back to ``args.<field>``."""
    nested = getattr(args, holder, None)
    if nested is not None and hasattr(nested, field):
        return getattr(nested, field)
    return getattr(args, field, None)


def unsupported_sampling_reasons(args: Any) -> list[str]:
    """Name every requested feature the speculative path would discard."""
    reasons: list[str] = []
    for holder, field, neutral in _NEUTRAL_PENALTIES:
        value = _resolve(args, holder, field)
        if value is None:
            continue
        try:
            if all(float(value) != float(n) for n in neutral):
                reasons.append(field)
        except (TypeError, ValueError):
            reasons.append(field)
    if _resolve(args, "logits", "logit_bias"):
        reasons.append("logit_bias")
    if getattr(args, "logprobs", False):
        reasons.append("logprobs")
    try:
        if int(getattr(args, "top_logprobs", -1) or -1) > 0:
            reasons.append("top_logprobs")
    except (TypeError, ValueError):
        pass
    return reasons


def fixed_arena_active() -> bool:
    """True when a preallocated Turbo4 arena owns the target KV this process."""
    if os.getenv("LOCAL_LLM_DFLASH_TURBOQUANT", "").strip().lower() != "turbo4":
        return False
    try:
        return int(os.getenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "0")) > 0
    except ValueError:
        return False


def install_exact_sampling_guard() -> None:
    """Send requests using dropped sampling features to exact target-only AR."""
    if os.getenv(_GUARD_ENV, "1").strip() != "1":
        return

    import mlx_lm.server as mlx_server
    from dflash_mlx import serve as dflash_serve

    generator = dflash_serve.DFlashResponseGenerator
    original = generator._serve_single
    if getattr(original, "_local_llm_exact_sampling", False):
        return
    parent_serve_single = mlx_server.ResponseGenerator._serve_single

    def _serve_single(self: Any, request: Any):
        try:
            rqueue, _, args = request
        except (TypeError, ValueError):
            return original(self, request)
        reasons = unsupported_sampling_reasons(args)
        if not reasons:
            return original(self, request)
        listed = ", ".join(sorted(reasons))

        if fixed_arena_active():
            # Exact decode runs through mlx-lm's own handler, which builds its
            # cache with make_prompt_cache() rather than the DFlash target ops
            # our fixed arena is attached to. That silently switches the target
            # to a full-precision KV cache outside the reservation -- merely
            # wasteful at 4K, but at 200K+ it is a different memory regime that
            # can exhaust the machine. Refuse instead.
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] rejected | "
                f"{listed} unsupported in fixed Turbo4 mode\n"
            )
            sys.stderr.flush()
            rqueue.put(
                ValueError(
                    f"{listed} cannot be honoured in fixed Turbo4 long-context "
                    "mode: exact decode would allocate a native KV cache "
                    "outside the reserved arena. Resend without these options, "
                    "or run a profile without a fixed arena."
                )
            )
            return None

        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] exact AR | "
            f"speculative decode ignores {listed}\n"
        )
        sys.stderr.flush()
        saved_draft_model = self.model_provider.draft_model
        try:
            self.model_provider.draft_model = None
            return parent_serve_single(self, request)
        finally:
            self.model_provider.draft_model = saved_draft_model

    _serve_single._local_llm_exact_sampling = True  # type: ignore[attr-defined]
    _serve_single._local_llm_original = original  # type: ignore[attr-defined]
    generator._serve_single = _serve_single
