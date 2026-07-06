import json
from pathlib import Path

import pandas as pd

from scripts.qwen3vl_cot_generation_audit import (
    analyze_generation,
    build_messages,
    summarize_records,
    write_summary_markdown,
)


def test_build_messages_replaces_image_placeholder_and_preserves_text(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    prompt = [
        {
            "role": "user",
            "content": "<image>\nPlease reason first.\nQuestion: choose A/B/C/D",
        }
    ]

    messages, prompt_text = build_messages(prompt, [str(image_path)])

    assert prompt_text == "Please reason first.\nQuestion: choose A/B/C/D"
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def test_analyze_generation_detects_thinking_and_early_final_answer():
    response = "<think>\nOption A seems right. Final answer: A\nMore reasoning\n</think>\nAnswer: A"

    result = analyze_generation(
        response,
        output_token_count=17,
        finish_reason="stop",
        rendered_prompt_tail="assistant\n",
    )

    assert result["starts_with_think"] is True
    assert result["has_think_open"] is True
    assert result["has_think_close"] is True
    assert result["early_final_answer"] is True
    assert result["final_answer_before_think_close"] is True
    assert result["truncated_by_length"] is False


def test_analyze_generation_counts_template_prefilled_think():
    result = analyze_generation(
        "Okay, let's reason step by step.",
        output_token_count=8,
        finish_reason="stop",
        rendered_prompt_tail="<|im_start|>assistant\n<think>\n",
    )

    assert result["template_prefilled_think"] is True
    assert result["combined_starts_with_think"] is True
    assert result["has_think_open"] is True
    assert result["starts_with_think"] is False


def test_summarize_records_reports_lengths_and_counts():
    records = [
        {
            "output_token_count": 100,
            "starts_with_think": True,
            "has_think_open": True,
            "has_think_close": False,
            "early_final_answer": False,
            "final_answer_before_think_close": False,
            "truncated_by_length": True,
            "finish_reason": "length",
        },
        {
            "output_token_count": 300,
            "starts_with_think": True,
            "has_think_open": True,
            "has_think_close": True,
            "early_final_answer": True,
            "final_answer_before_think_close": True,
            "truncated_by_length": False,
            "finish_reason": "stop",
        },
    ]

    summary = summarize_records(records)

    assert summary["num_records"] == 2
    assert summary["output_tokens_mean"] == 200
    assert summary["output_tokens_min"] == 100
    assert summary["output_tokens_max"] == 300
    assert summary["starts_with_think_count"] == 2
    assert summary["has_think_close_count"] == 1
    assert summary["truncated_by_length_count"] == 1
    assert summary["early_final_answer_count"] == 1


def test_write_summary_markdown_includes_sample_preview(tmp_path):
    summary = {
        "num_records": 1,
        "output_tokens_mean": 42,
        "output_tokens_min": 42,
        "output_tokens_p50": 42,
        "output_tokens_p90": 42,
        "output_tokens_max": 42,
        "starts_with_think_count": 1,
        "has_think_open_count": 1,
        "has_think_close_count": 0,
        "early_final_answer_count": 0,
        "final_answer_before_think_close_count": 0,
        "truncated_by_length_count": 0,
        "finish_reason_counts": {"stop": 1},
    }
    records = [
        {
            "index": 0,
            "question_id": "q1",
            "answer": "A",
            "output_token_count": 42,
            "response": "<think>\nreasoning",
        }
    ]
    out = tmp_path / "summary.md"

    write_summary_markdown(out, summary, records)

    text = out.read_text(encoding="utf-8")
    assert "Qwen3-VL CoT Generation Audit" in text
    assert "q1" in text
    assert "<think>" in text
