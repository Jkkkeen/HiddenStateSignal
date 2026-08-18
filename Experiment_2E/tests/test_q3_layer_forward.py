from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiment_2e.q3_layer_forward import (
    pool_hidden_states_device,
    teacher_forced_pooled_forward,
    tokenize_response,
)
from experiment_2e.reduction import pool_token_hidden, trajectory_endpoints


class RoundTripTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(int(value)) for value in ids)


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        input_ids,
        use_cache,
        output_hidden_states,
        return_dict,
        logits_to_keep,
    ):
        assert use_cache is False
        assert output_hidden_states is True
        assert return_dict is True
        sequence = input_ids.shape[1]
        base = torch.arange(sequence * 4, device=input_ids.device, dtype=torch.float32).reshape(
            1, sequence, 4
        )
        hidden_states = (base, base + 1.0, base + 2.0)
        vocab = 256
        logits = torch.zeros((1, sequence, vocab), device=input_ids.device)
        logits.scatter_(2, input_ids[:, :, None], 2.0)
        return SimpleNamespace(
            hidden_states=hidden_states,
            logits=logits[:, logits_to_keep, :],
        )


def test_tokenize_response_preserves_separate_boundary():
    prompt_ids, response_ids = tokenize_response(RoundTripTokenizer(), "ab", " cd")
    assert prompt_ids == [97, 98]
    assert response_ids == [32, 99, 100]


def test_tokenize_response_rejects_non_roundtrip():
    tokenizer = RoundTripTokenizer()
    tokenizer.decode = lambda *args, **kwargs: "different"
    with pytest.raises(ValueError, match="round-trip"):
        tokenize_response(tokenizer, "ab", "cd")


def test_device_pool_matches_numpy_reference():
    token_hidden = np.arange(9 * 3 * 4, dtype=np.float32).reshape(9, 3, 4)
    states = tuple(torch.tensor(token_hidden[:, layer, :])[None, :, :] for layer in range(3))
    endpoints = trajectory_endpoints(9, window=4, stride=2)
    actual = pool_hidden_states_device(
        states,
        start=0,
        stop=9,
        endpoints=endpoints,
        window=4,
    )
    expected = pool_token_hidden(token_hidden, endpoints, window=4)
    assert np.allclose(actual["mean_w128_s32"], expected["mean_w128_s32"])
    assert np.allclose(actual["last_s32"], expected["last_s32"])


def test_teacher_forced_forward_returns_only_pooled_hidden_and_controls():
    result = teacher_forced_pooled_forward(
        FakeCausalLM(),
        RoundTripTokenizer(),
        "ab",
        "cdefg",
        window=4,
        stride=2,
    )
    assert result["prompt_token_count"] == 2
    assert result["response_token_count"] == 5
    assert result["n_hidden_states"] == 3
    assert result["mean_w128_s32"].shape == (2, 3, 4)
    assert result["last_s32"].shape == (2, 3, 4)
    assert result["token_logprob"].shape == (5,)
    assert result["policy_entropy"].shape == (5,)
    assert result["hidden_norm"].shape == (5,)
    assert np.isfinite(result["token_logprob"]).all()
    assert np.isfinite(result["policy_entropy"]).all()
    assert "token_hidden" not in result
