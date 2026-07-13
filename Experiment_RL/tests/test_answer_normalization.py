import pytest

from scripts.answer_normalization import (
    accepted_mcq_surface_forms,
    accepted_surface_forms,
    normalize_mcq_content,
    normalize_semantic_answer,
    semantic_equivalent,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A:  100   degrees", "100 degrees"),
        ("B) 4 sqrt(2)", "4 sqrt(2)"),
        ("C.\n  1/2", "1/2"),
    ],
)
def test_normalize_mcq_content_removes_label_and_collapses_whitespace(raw, expected):
    assert normalize_mcq_content(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000", "1000"),
        ("+0007", "7"),
        ("2/4", "1/2"),
        ("0.50", "1/2"),
    ],
)
def test_normalize_semantic_answer_canonicalizes_exact_numbers(raw, expected):
    assert normalize_semantic_answer(raw).canonical == expected


def test_accepted_surface_forms_are_deterministic_for_exact_rational():
    assert accepted_surface_forms("2/4") == ("1/2", "\\frac{1}{2}", "0.5")


def test_percentage_requires_explicit_compatibility():
    with pytest.raises(ValueError, match="percentage"):
        normalize_semantic_answer("50%")

    answer = normalize_semantic_answer("50%", allow_percentage=True)
    assert answer.canonical == "1/2"
    assert accepted_surface_forms("50%", allow_percentage=True) == (
        "1/2",
        "\\frac{1}{2}",
        "0.5",
        "50%",
    )
    assert semantic_equivalent("50%", "1/2", allow_percentage=True)


def test_radical_tuple_and_set_have_stable_semantics():
    radical = normalize_semantic_answer("sqrt(20)")
    assert radical.canonical == "2*sqrt(5)"
    assert semantic_equivalent("sqrt(20)", "2*sqrt(5)")

    ordered = normalize_semantic_answer("(1/2, 2)")
    assert ordered.kind == "tuple"
    assert ordered.canonical == "(1/2, 2)"
    assert not semantic_equivalent("(1/2, 2)", "(2, 1/2)")

    unordered = normalize_semantic_answer("{2, 1/2}")
    assert unordered.kind == "set"
    assert unordered.canonical == "{1/2, 2}"
    assert semantic_equivalent("{2, 1/2}", "{1/2, 2}")


def test_mcq_surface_forms_preserve_original_and_expand_exact_semantics():
    fraction = accepted_mcq_surface_forms("B: 2/4")
    radical = accepted_mcq_surface_forms("4 sqrt(2)")
    unit = accepted_mcq_surface_forms("0.50 cm")
    squared_unit = accepted_mcq_surface_forms("0.50 cm²")
    unsupported = accepted_mcq_surface_forms("the shaded region")

    assert fraction.forms == ("2/4", "1/2", "\\frac{1}{2}", "0.5")
    assert fraction.status == "parsed_semantic"
    assert radical.forms == ("4 sqrt(2)", "4*sqrt(2)")
    assert radical.status == "parsed_semantic"
    assert unit.forms == ("0.50 cm", "1/2 cm", "\\frac{1}{2} cm", "0.5 cm")
    assert unit.status == "parsed_unit"
    assert squared_unit.forms == (
        "0.50 cm²",
        "1/2 cm²",
        "\\frac{1}{2} cm²",
        "0.5 cm²",
    )
    assert squared_unit.status == "parsed_unit"
    assert unsupported.forms == ("the shaded region",)
    assert unsupported.status == "canonical_only_unsupported"
