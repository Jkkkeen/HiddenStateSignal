from scripts.response_segment_steps import (
    answer_after_think,
    split_response_steps,
)


def test_answer_after_think_returns_text_after_close_tag():
    response = "<think>hidden reasoning</think>\n\nFinal answer is B.\nB"

    segment, start, end, status = answer_after_think(response)

    assert segment == "Final answer is B.\nB"
    assert response[start:end].strip() == segment
    assert status == "after_close"


def test_split_response_steps_prefers_step_headings_over_sentences():
    text = """### Step 1: Setup
Use similar triangles. The ratio is fixed.

### Step 2: Solve
Compute h = 8.

### Final Answer
The answer is C.
"""

    steps, mode = split_response_steps(text)

    assert mode == "block"
    assert [step["heading"] for step in steps] == [
        "### Step 1: Setup",
        "### Step 2: Solve",
        "### Final Answer",
    ]
    assert "The ratio is fixed." in steps[0]["text"]
    assert "The answer is C." in steps[-1]["text"]


def test_split_response_steps_detects_bold_markdown_step_headings():
    text = """To solve the problem, inspect the figure.

### **Step 1: Understand the Triangle Configuration**
- Triangle ABC is isosceles.
- It is right-angled at C.

### **Step 2: Use Key Geometric Properties**
Use the angle bisector property.

### **Final Answer**
The answer is A.
"""

    steps, mode = split_response_steps(text)

    assert mode == "block"
    assert len(steps) == 3
    assert steps[0]["heading"] == "### **Step 1: Understand the Triangle Configuration**"
    assert "Triangle ABC is isosceles" in steps[0]["text"]
    assert steps[-1]["heading"] == "### **Final Answer**"


def test_split_response_steps_uses_numbered_blocks():
    text = """1. First find the radius.
It is 9.
2. Then apply Pythagoras.
The result is D."""

    steps, mode = split_response_steps(text)

    assert mode == "block"
    assert len(steps) == 2
    assert steps[0]["heading"] == "1. First find the radius."
    assert steps[1]["heading"] == "2. Then apply Pythagoras."


def test_split_response_steps_falls_back_to_sentences_without_headings():
    text = "Use similar triangles. Set h over 10 equal to 1.6 over 2. Therefore h is 8."

    steps, mode = split_response_steps(text)

    assert mode == "sentence"
    assert len(steps) == 3
    assert steps[0]["text"] == "Use similar triangles."


def test_split_response_steps_marks_short_answer_as_terminal_pair():
    steps, mode = split_response_steps("B")

    assert mode == "terminal"
    assert len(steps) == 1
    assert steps[0]["text"] == "B"
