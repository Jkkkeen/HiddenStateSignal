from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_semantic_step_basin_qwen3vl import (
    char_token_span,
    select_response_segment_steps,
)


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def test_select_response_segment_steps_answer_prefers_block_steps() -> None:
    response = """<think>private reasoning</think>

### **Step 1: Setup**
Use similar triangles.

### **Step 2: Solve**
Compute h = 8.

### **Final Answer**
The answer is C.
"""

    selected = select_response_segment_steps(
        response,
        segment="answer",
        step_mode="auto",
        min_step_chars=12,
    )

    assert selected["segment_text"].startswith("### **Step 1")
    assert selected["text_segment_status"] == "after_close"
    assert selected["step_mode_used"] == "block"
    assert [step["heading"] for step in selected["steps"]] == [
        "### **Step 1: Setup**",
        "### **Step 2: Solve**",
        "### **Final Answer**",
    ]


def test_char_token_span_uses_stripped_answer_offsets() -> None:
    tokenizer = WhitespaceTokenizer()
    response = "<think> a b </think>\n\n  answer starts here"
    start = response.index("answer")
    end = len(response)

    token_start, token_end, status = char_token_span(tokenizer, response, start, end)

    assert status == "char_span"
    assert token_start == len(response[:start].split())
    assert token_end == len(response[:end].split())
    assert token_end > token_start
