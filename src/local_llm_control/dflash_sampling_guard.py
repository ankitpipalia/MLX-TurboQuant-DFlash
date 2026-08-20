"""Honour sampling parameters DFlash's speculative path silently drops.

``serve.py`` parses ``repetition_penalty``, ``presence_penalty``,
``frequency_penalty``, ``xtc_probability``, ``xtc_threshold``, ``logit_bias``,
``logprobs`` and ``top_logprobs`` off the request body, but the runtime forwards
only ``temperature``/``top_p``/``top_k``/``min_p`` and
``stream_dflash_generate_impl`` accepts only those four. Everything else is
therefore accepted and quietly ignored -- not rejected, not applied -- so a
client asking for a repetition penalty or logprobs gets output that silently
disregards it.

Rather than reimplement those features inside the speculative loop, route such
requests to plain autoregressive decode on the target, which is mlx-lm's own
server path and does honour them. That is exactly the mechanism the upstream
``--fastpath-max-tokens`` option already uses: drop the draft model for the
duration of the request and delegate to the parent handler.

Note that temperature, top_p, top_k and min_p *are* forwarded, so ordinary
non-greedy sampling still gets the speculative path.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_GUARD_ENV = "LOCAL_LLM_DFLASH_EXACT_SAMPLING"

# ``serve.py`` defaults these to 0.0, so 0.0 means "unset". A repetition penalty
# of 1.0 is the conventional no-op and is treated as unset too, so a client
# sending the neutral value does not needlessly lose speculative decoding.
_NEUTRAL_PENALTIES = {
    "repetition_penalty": (0.0, 1.0),
    "presence_penalty": (0.0,),
    "frequency_penalty": (0.0,),
    "xtc_probability": (0.0,),
}


def unsupported_sampling_reasons(args: Any) -> list[str]:
    """Name every requested feature the speculative path would discard."""
    reasons: list[str] = []
    for field, neutral in _NEUTRAL_PENALTIES.items():
        value = getattr(args, field, None)
        if value is None:
            continue
        try:
            if all(float(value) != float(n) for n in neutral):
                reasons.append(field)
        except (TypeError, ValueError):
            reasons.append(field)
    if getattr(args, "logit_bias", None):
        reasons.append("logit_bias")
    if getattr(args, "logprobs", False):
        reasons.append("logprobs")
    try:
        if int(getattr(args, "top_logprobs", -1) or -1) > 0:
            reasons.append("top_logprobs")
    except (TypeError, ValueError):
        pass
    return reasons


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
            _, _, args = request
        except (TypeError, ValueError):
            return original(self, request)
        reasons = unsupported_sampling_reasons(args)
        if not reasons:
            return original(self, request)

        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] exact AR | "
            f"speculative decode ignores {', '.join(sorted(reasons))}\n"
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
