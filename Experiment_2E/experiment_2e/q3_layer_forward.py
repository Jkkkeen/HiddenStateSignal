from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .reduction import trajectory_endpoints


def tokenize_response(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    decoded = tokenizer.decode(
        response_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not prompt_ids or not response_ids:
        raise ValueError("prompt and response must both contain tokens")
    if decoded != response:
        raise ValueError("response token round-trip mismatch")
    return prompt_ids, response_ids


def pool_hidden_states_device(
    hidden_states: Iterable[Any],
    *,
    start: int,
    stop: int,
    endpoints: np.ndarray,
    window: int = 128,
) -> dict[str, np.ndarray]:
    import torch

    layers = tuple(hidden_states)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if not layers:
        raise ValueError("model returned no hidden states")
    if start < 0 or stop <= start or endpoints.size < 1:
        raise ValueError("invalid response boundary or endpoints")
    if np.any(endpoints < 1) or np.any(endpoints > stop - start):
        raise ValueError("trajectory endpoint falls outside response tokens")

    endpoint_tensor = torch.as_tensor(endpoints, dtype=torch.long)
    means = []
    lasts = []
    for state in layers:
        if state.ndim != 3 or state.shape[0] != 1 or stop > state.shape[1]:
            raise ValueError(f"unsupported hidden-state shape: {tuple(state.shape)}")
        response = state[0, start:stop].detach().float()
        device_endpoints = endpoint_tensor.to(response.device)
        starts = torch.clamp(device_endpoints - window, min=0)
        prefix = torch.cat(
            [
                torch.zeros(
                    (1, response.shape[1]),
                    dtype=response.dtype,
                    device=response.device,
                ),
                response.cumsum(dim=0),
            ],
            dim=0,
        )
        denominator = (device_endpoints - starts).to(response.dtype)[:, None]
        means.append(((prefix[device_endpoints] - prefix[starts]) / denominator).cpu())
        lasts.append(response[device_endpoints - 1].cpu())
    return {
        "mean_w128_s32": torch.stack(means, dim=1).numpy(),
        "last_s32": torch.stack(lasts, dim=1).numpy(),
    }


def teacher_forced_pooled_forward(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    *,
    window: int = 128,
    stride: int = 32,
) -> dict[str, Any]:
    import torch

    prompt_ids, response_ids = tokenize_response(tokenizer, prompt, response)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    start = len(prompt_ids)
    stop = start + len(response_ids)
    prediction_positions = torch.arange(start - 1, stop - 1, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=prediction_positions,
        )
    if output.hidden_states is None:
        raise ValueError("model forward did not return hidden states")
    endpoints = trajectory_endpoints(len(response_ids), window=window, stride=stride)
    pooled = pool_hidden_states_device(
        output.hidden_states,
        start=start,
        stop=stop,
        endpoints=endpoints,
        window=window,
    )

    logits = output.logits[0].float()
    if logits.shape[0] != len(response_ids):
        raise ValueError(
            f"expected {len(response_ids)} response logits, found {logits.shape[0]}"
        )
    targets = torch.tensor(response_ids, dtype=torch.long, device=device)
    log_normalizer = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    policy_entropy = log_normalizer - (probabilities * logits).sum(dim=-1)
    token_logprob = logits.gather(1, targets[:, None]).squeeze(1) - log_normalizer
    final_hidden = output.hidden_states[-1][0, start:stop].detach().float()
    hidden_norm = torch.linalg.vector_norm(final_hidden, dim=1)

    return {
        **pooled,
        "endpoints": endpoints,
        "progress": endpoints.astype(np.float32) / len(response_ids),
        "token_logprob": token_logprob.cpu().numpy(),
        "policy_entropy": policy_entropy.cpu().numpy(),
        "hidden_norm": hidden_norm.cpu().numpy(),
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "n_hidden_states": len(output.hidden_states),
    }
