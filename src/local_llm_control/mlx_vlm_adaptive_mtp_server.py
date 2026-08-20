"""Run Qwen native MTP with Yukon's acceptance-cost depth policy.

The upstream Python runtime already preserves committed Qwen MTP history and
performs exact target verification.  Its Qwen drafter, however, treats the CLI
block size as a fixed order.  Yukon treats it as a ceiling and chooses a depth
from conditional per-position acceptance EMAs.  This wrapper ports that part
of the policy without modifying the pinned mlx-vlm installation.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .mlx_vlm_text_server import (
    install_reasoning_strength_bridge,
    install_text_first_loader,
)


def adaptive_mtp_block_total(
    requested_block_total: int,
    accept_lens: list[int],
    draft_lens: list[int],
    *,
    head_cost_ratio: float = 0.18,
) -> int:
    """Return primary-plus-draft width using Yukon's marginal cost rule.

    Python mlx-vlm currently needs at least one draft token, so the policy's
    free serial-skip case is represented by depth one.  The requested width is
    a ceiling; the Qwen target still verifies every proposed token.
    """
    max_depth = max(1, int(requested_block_total) - 1)
    probabilities = [0.85 * math.pow(0.98, i) for i in range(max_depth)]
    alpha = 0.15

    for accepted, drafted in zip(accept_lens, draft_lens):
        accepted = max(0, min(int(accepted), max_depth))
        drafted = max(0, min(int(drafted), max_depth))
        for index in range(min(accepted, len(probabilities))):
            probabilities[index] += alpha * (1.0 - probabilities[index])
        if accepted < drafted and accepted < len(probabilities):
            probabilities[accepted] += alpha * (0.0 - probabilities[accepted])
        elif accepted == drafted and drafted and accepted < len(probabilities):
            probabilities[accepted] += alpha * (0.95 - probabilities[accepted])

    reach = 1.0
    expected = 0.0
    depth = 0
    h = max(0.01, float(head_cost_ratio))
    while depth < max_depth:
        reach *= probabilities[depth]
        threshold = h * (1.0 + expected) / (1.0 + depth * h)
        if reach <= threshold:
            break
        expected += reach
        depth += 1
    return max(2, depth + 1)


def install_adaptive_mtp_policy() -> None:
    from mlx_vlm.speculative import mtp

    original = mtp._mtp_next_block_size
    if getattr(original, "_local_llm_adaptive_mtp", False):
        return

    def adaptive(
        draft_model: Any,
        requested_block_total: int,
        configured_block_total: int,
        remaining_budget: int,
    ) -> int:
        model_type = getattr(getattr(draft_model, "config", None), "model_type", "")
        if model_type != "qwen3_5_mtp":
            return original(
                draft_model,
                requested_block_total,
                configured_block_total,
                remaining_budget,
            )
        ceiling = min(int(requested_block_total), int(remaining_budget))
        if ceiling <= 2:
            return ceiling
        ratio = float(os.getenv("LOCAL_LLM_MTP_HEAD_COST_RATIO", "0.18"))
        return min(
            ceiling,
            adaptive_mtp_block_total(
                ceiling,
                list(getattr(draft_model, "accept_lens", []) or []),
                list(getattr(draft_model, "draft_lens", []) or []),
                head_cost_ratio=ratio,
            ),
        )

    adaptive._local_llm_adaptive_mtp = True  # type: ignore[attr-defined]
    adaptive._local_llm_original = original  # type: ignore[attr-defined]
    mtp._mtp_next_block_size = adaptive


def main() -> None:
    install_text_first_loader()
    install_reasoning_strength_bridge()
    install_adaptive_mtp_policy()
    from mlx_vlm.server.cli import main as server_main

    server_main()


if __name__ == "__main__":
    main()
