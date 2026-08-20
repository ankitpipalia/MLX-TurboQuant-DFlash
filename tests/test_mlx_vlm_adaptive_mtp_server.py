from local_llm_control.mlx_vlm_adaptive_mtp_server import (
    adaptive_mtp_block_total,
)


def test_adaptive_mtp_starts_below_ceiling() -> None:
    assert adaptive_mtp_block_total(6, [], []) == 5


def test_adaptive_mtp_widens_after_full_acceptance() -> None:
    accepted = [4] * 12
    drafted = [4] * 12
    assert adaptive_mtp_block_total(6, accepted, drafted) == 6


def test_adaptive_mtp_contracts_after_rejections() -> None:
    accepted = [0] * 12
    drafted = [4] * 12
    assert adaptive_mtp_block_total(6, accepted, drafted) == 2


def test_adaptive_mtp_never_exceeds_request() -> None:
    assert adaptive_mtp_block_total(3, [8] * 20, [8] * 20) <= 3
