#!/usr/bin/env python3
"""Deterministic answer normalization and surface-form generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)


CHOICE_PREFIX = re.compile(r"^\s*[A-Z]\s*(?::|\)|\.)\s*", re.IGNORECASE)
CHOICE_LINE = re.compile(r"^\s*([A-Z])\s*(?::|\)|\.)\s*(.*?)\s*$")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+\s*/\s*[+-]?\d+)$")
SYMBOLIC = re.compile(r"^[A-Za-z0-9_+\-*/^().\s]+$")
PARSE_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    rationalize,
    implicit_multiplication_application,
)
UNIT_SUFFIX = re.compile(
    r"^(.+?)\s+(degrees?|cm|mm|m|km|in|ft|kg|g|seconds?|minutes?|hours?)"
    r"(?:\^([23])|([\u00b2\u00b3]))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticAnswer:
    kind: str
    canonical: str
    value: Any


@dataclass(frozen=True)
class McqSurfaceForms:
    forms: tuple[str, ...]
    status: str


def normalize_mcq_content(value: str) -> str:
    """Remove an option label and normalize whitespace without changing content."""

    content = CHOICE_PREFIX.sub("", str(value or ""), count=1)
    content = " ".join(content.split())
    if not content:
        raise ValueError("MCQ option content must be non-empty")
    return content


def parse_mcq_choices(prompt: str) -> dict[str, str]:
    """Parse contiguous one-line choices beginning at label A."""

    choices: dict[str, str] = {}
    for line in str(prompt or "").splitlines():
        match = CHOICE_LINE.match(line)
        if match and match.group(2):
            choices[match.group(1).upper()] = match.group(2).strip()
    labels = sorted(choices)
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if len(labels) < 2 or labels != expected:
        raise ValueError("choices must use at least two contiguous labels starting at A")
    return {label: choices[label] for label in labels}


def _canonical_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "({[":
            depth += 1
        elif char in ")} ]".replace(" ", ""):
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    if any(not part for part in parts):
        raise ValueError("structured answer contains an empty item")
    return parts


def normalize_semantic_answer(
    value: str,
    *,
    allow_percentage: bool = False,
) -> SemanticAnswer:
    """Parse a supported free-form answer into an exact semantic object."""

    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        items = [
            normalize_semantic_answer(item, allow_percentage=allow_percentage)
            for item in _split_top_level(text[1:-1])
        ]
        return SemanticAnswer(
            "tuple",
            "(" + ", ".join(item.canonical for item in items) + ")",
            tuple(item.value for item in items),
        )
    if len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        items = [
            normalize_semantic_answer(item, allow_percentage=allow_percentage)
            for item in _split_top_level(text[1:-1])
        ]
        items.sort(key=lambda item: item.canonical)
        return SemanticAnswer(
            "set",
            "{" + ", ".join(item.canonical for item in items) + "}",
            frozenset(item.value for item in items),
        )

    raw = "".join(text.split()).replace(",", "")
    is_percentage = raw.endswith("%")
    if is_percentage:
        if not allow_percentage:
            raise ValueError("percentage answer requires explicit compatibility")
        raw = raw[:-1]
    if raw and NUMBER.fullmatch(raw):
        number = Fraction(raw)
        if is_percentage:
            number /= 100
        return SemanticAnswer("number", _canonical_fraction(number), number)
    if not raw or "__" in raw or not SYMBOLIC.fullmatch(raw):
        raise ValueError(f"unsupported answer format: {value!r}")
    try:
        expression = parse_expr(
            raw,
            local_dict={"sqrt": sympy.sqrt},
            transformations=PARSE_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as error:
        raise ValueError(f"unsupported answer format: {value!r}") from error
    expression = sympy.simplify(expression)
    if expression.free_symbols:
        raise ValueError(f"unsupported symbolic names in answer: {value!r}")
    return SemanticAnswer("expression", sympy.sstr(expression), expression)


def _finite_decimal(value: Fraction) -> str | None:
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return None
    with localcontext() as context:
        context.prec = max(32, len(str(abs(value.numerator))) + len(str(value.denominator)) + 4)
        rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def accepted_surface_forms(
    value: str,
    *,
    allow_percentage: bool = False,
) -> tuple[str, ...]:
    """Return deterministic finite renderings for a supported semantic answer."""

    answer = normalize_semantic_answer(value, allow_percentage=allow_percentage)
    if answer.kind != "number":
        return (answer.canonical,)
    number = answer.value
    surfaces = [answer.canonical]
    if number.denominator != 1:
        surfaces.append(f"\\frac{{{number.numerator}}}{{{number.denominator}}}")
        decimal = _finite_decimal(number)
        if decimal is not None:
            surfaces.append(decimal)
    if allow_percentage and str(value).strip().endswith("%"):
        percentage = _finite_decimal(number * 100)
        if percentage is not None:
            surfaces.append(f"{percentage}%")
    return tuple(dict.fromkeys(surfaces))


def accepted_mcq_surface_forms(value: str) -> McqSurfaceForms:
    """Expand only deterministic semantic variants while preserving MCQ content."""

    content = normalize_mcq_content(value)
    unit_match = UNIT_SUFFIX.fullmatch(content)
    if unit_match:
        scalar = unit_match.group(1)
        unit = content[len(scalar) :].strip()
        try:
            scalar_forms = accepted_surface_forms(scalar)
        except ValueError:
            return McqSurfaceForms((content,), "canonical_only_unsupported")
        forms = [content, *(f"{surface} {unit}" for surface in scalar_forms)]
        return McqSurfaceForms(tuple(dict.fromkeys(forms)), "parsed_unit")
    try:
        semantic_forms = accepted_surface_forms(content)
    except ValueError:
        return McqSurfaceForms((content,), "canonical_only_unsupported")
    return McqSurfaceForms(
        tuple(dict.fromkeys((content, *semantic_forms))),
        "parsed_semantic",
    )


def semantic_equivalent(
    left: str,
    right: str,
    *,
    allow_percentage: bool = False,
) -> bool:
    """Compare two supported answers through their parsed semantic values."""

    return normalize_semantic_answer(
        left, allow_percentage=allow_percentage
    ).value == normalize_semantic_answer(right, allow_percentage=allow_percentage).value
