from scripts.answer_likelihood_scoring import (
    ANSWER_END,
    DENSE_THINK_FRACS,
    INTERFACES,
    PROTOCOL_VERSION,
    SPARSE_THINK_FRACS,
    build_assistant_scoring_text,
    summarize_surface_trajectory,
    target_span_from_rendered,
)


def test_protocol_constants_freeze_interfaces_terminator_and_probe_grids():
    assert PROTOCOL_VERSION == "rl03_stage_a_v2"
    assert INTERFACES == {
        "I0": "\n\nGiven the reasoning so far, the answer is (",
        "I1": "\n\nFinal answer: ",
        "I2": "\n\nGiven the reasoning so far, the final answer is: ",
    }
    assert ANSWER_END == "\n"
    assert SPARSE_THINK_FRACS == (0.0, 0.25, 0.50, 0.90)
    assert DENSE_THINK_FRACS == (
        0.05,
        0.10,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    )


def test_build_assistant_scoring_text_marks_only_target_and_terminator():
    rendered, start, end = build_assistant_scoring_text(
        reasoning_prefix="the value 1/2 appears in the work",
        interface=INTERFACES["I1"],
        target="1/2",
    )

    assert rendered == "the value 1/2 appears in the work\n\nFinal answer: 1/2\n"
    assert rendered[start:end] == "1/2\n"
    assert start == rendered.rfind("1/2")


def test_target_span_from_rendered_uses_final_interface_occurrence():
    rendered = (
        "Question contains Final answer: 1/2\n"
        "<assistant>work mentions 1/2\n\nFinal answer: 1/2\n<end>"
    )

    start, end = target_span_from_rendered(rendered, INTERFACES["I1"], "1/2")

    assert rendered[start:end] == "1/2\n"
    assert start == rendered.rfind("1/2")


def test_surface_trajectory_uses_median_of_gains_and_retains_adjacent_rates():
    scores = {
        "canonical": {0.0: 0.0, 0.25: 0.0, 0.50: 50.0, 0.90: 100.0, 1.0: 101.0, "trimmed": 90.0},
        "latex": {0.0: 0.0, 0.25: 10.0, 0.50: 10.5, 0.90: 11.0, 1.0: 12.0, "trimmed": 10.0},
        "decimal": {0.0: 0.0, 0.25: 20.0, 0.50: 16.0, 0.90: 12.0, 1.0: 11.0, "trimmed": 13.0},
    }

    result = summarize_surface_trajectory(scores)

    assert result["level_by_probe"][0.25] == 10.0
    assert result["level_by_probe"][0.90] == 12.0
    assert result["gold_gain_25_90"] == 1.0
    assert result["gold_gain_25_90"] != 12.0 - 10.0
    assert result["gold_gain_25_trimmed"] == 0.0
    assert result["step_gain_by_interval"][(0.25, 0.50)] == 0.5
    assert result["step_gain_rate_by_interval"][(0.25, 0.50)] == 2.0
    assert result["step_gain_rate_by_interval"][(0.50, 0.90)] == 1.25
    assert all("trimmed" not in interval for interval in result["step_gain_by_interval"])


def test_surface_trajectory_reports_non_telescope_shape_summaries():
    result = summarize_surface_trajectory(
        {"canonical": {0.0: 0.0, 0.25: 0.0, 0.50: 1.0, 0.90: 0.5, 1.0: 0.6, "trimmed": 0.4}}
    )

    assert result["gold_positive_gain_rate"] == 0.5
    assert result["gold_monotonicity_violation_rate"] == 0.5
    assert isinstance(result["gold_trajectory_slope"], float)
