from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_c2_vllm_probe_default_logprobs_respects_vllm_cap() -> None:
    trainer_base = ROOT / "server_snapshots" / "c2_1_edit" / "verl" / "trainer" / "ppo" / "v1" / "trainer_base.py"

    text = trainer_base.read_text(encoding="utf-8")

    assert 'OPTION_GAIN_VLLM_LOGPROBS", "20"' in text


def test_c2_vllm_probe_defaults_to_prompt_logprob_tail_scoring() -> None:
    trainer_base = ROOT / "server_snapshots" / "c2_1_edit" / "verl" / "trainer" / "ppo" / "v1" / "trainer_base.py"

    text = trainer_base.read_text(encoding="utf-8")

    assert 'OPTION_GAIN_VLLM_PROBE_METHOD", "prompt_logprobs_tail"' in text
    assert 'OPTION_GAIN_LABEL_TAIL", ")"' in text
    assert 'OPTION_GAIN_VLLM_PROMPT_LOGPROBS", "20"' in text


def test_c2_actor_forward_probe_is_wired_into_fsdp_snapshot() -> None:
    transformer_impl = (
        ROOT
        / "server_snapshots"
        / "c2_1_edit"
        / "verl"
        / "workers"
        / "engine"
        / "fsdp"
        / "transformer_impl.py"
    )

    text = transformer_impl.read_text(encoding="utf-8")

    assert "c2_actor_option_logits" in text
    assert "c2_option_label_token_ids" in text
    assert "probe_positions_for_lengths" in text


def test_c2_actor_forward_probe_is_collected_in_old_log_prob_snapshot() -> None:
    trainer_base = ROOT / "server_snapshots" / "c2_1_edit" / "verl" / "trainer" / "ppo" / "v1" / "trainer_base.py"

    text = trainer_base.read_text(encoding="utf-8")

    assert "OPTION_GAIN_BACKEND" in text
    assert "actor_forward" in text
    assert "_compute_c2_actor_option_gain" in text
    assert "c2_actor_option_logits" in text
    assert "torch.nested.to_padded_tensor" in text
