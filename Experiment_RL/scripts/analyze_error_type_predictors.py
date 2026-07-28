#!/usr/bin/env python3
'''Analyze low-cost error-type proxies against revision recoverability.'''

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_recoverability import analyze_feature


MARKERS = {
    'self_correction': (r'\bwait\b', r'\bhowever\b', r'\bactually\b', r'\breconsider\b', r'\bcorrection\b', r'\bon second thought\b'),
    'visual_reference': (r'\bimage\b', r'\bdiagram\b', r'\bgraph\b', r'\bfigure\b', r'\bshown\b', r'\bvisual\b'),
    'option_mapping': (r'\boption\b', r'\bchoice\b', r'\bcorresponds? to\b', r'\bmatches?\b', r'\bfinal answer\b'),
}

PREDICTOR_DIRECTIONS = {
    'explicit_wrong_commitment': 'neg',
    'self_correction_per_1k_tokens': 'pos',
    'visual_reference_per_1k_tokens': 'pos',
    'option_mapping_per_1k_tokens': 'pos',
}

YES_NO_FIELDS = (
    'reasoning_value_correct',
    'selected_option_mapping_correct',
    'visual_reread_likely_helpful',
)


def _count_markers(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def add_text_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prefixes = result.get('revision_prefix', pd.Series('', index=result.index)).fillna('').astype(str)
    token_count = prefixes.str.findall(r'\S+').str.len().clip(lower=1)
    result['prefix_token_count'] = token_count
    claim = result.get('original_prediction_claim_flag', pd.Series(False, index=result.index))
    result['explicit_wrong_commitment'] = claim.fillna(False).astype(bool).astype(float)
    for name, patterns in MARKERS.items():
        count_column = f'{name}_marker_count' if name == 'self_correction' else f'{name}_count'
        result[count_column] = prefixes.map(lambda text: _count_markers(text, patterns))
        result[f'{name}_per_1k_tokens'] = result[count_column] * 1000.0 / token_count
    return result


def add_annotation_predictors(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    result = frame.copy()
    directions: dict[str, str] = {}
    if 'error_type' in result:
        labels = result['error_type'].fillna('').astype(str).str.strip().str.lower()
        for label in sorted(value for value in labels.unique() if value):
            column = f'error_type__{label}'
            result[column] = (labels == label).astype(float)
            directions[column] = 'pos'
    if 'commitment_level' in result:
        result['commitment_level'] = pd.to_numeric(result['commitment_level'], errors='coerce')
        if result['commitment_level'].notna().any():
            directions['commitment_level'] = 'neg'
    for field in YES_NO_FIELDS:
        if field not in result:
            continue
        values = result[field].fillna('').astype(str).str.strip().str.lower()
        column = f'{field}__yes'
        result[column] = values.map({'yes': 1.0, 'no': 0.0})
        if result[column].notna().any():
            directions[column] = 'pos'
    return result, directions


def analyze_predictor(
    frame: pd.DataFrame,
    predictor: str,
    direction: str,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prepared = frame.copy()
    if 'any_recovery' not in prepared:
        prepared['any_recovery'] = prepared['recovery_rate'].astype(float) > 0
    return analyze_feature(prepared, feature=predictor, direction=direction, n_boot=n_boot, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--annotations', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260711)
    return parser.parse_args()


def _load_manifest(path: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()])


def merge_manifest_scores(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keys = ['question_id', 'rollout_id']
    manifest_columns = keys + ['revision_prefix']
    if 'original_prediction_claim_flag' not in scores:
        manifest_columns.append('original_prediction_claim_flag')
    merged = scores.merge(manifest[manifest_columns], on=keys, how='inner', validate='one_to_one')
    if 'original_prediction_claim_flag' not in merged:
        merged['original_prediction_claim_flag'] = False
    return merged


def _write_report(path: Path, merged: pd.DataFrame, summary: pd.DataFrame) -> None:
    anchored = merged[merged['explicit_wrong_commitment'] == 1]
    unanchored = merged[merged['explicit_wrong_commitment'] == 0]
    lines = [
        '# Error-Type Predictor Discovery', '',
        'This is exploratory analysis on the same discovery-smoke questions. Text proxies are not semantic error labels and cannot support a confirmatory claim.', '',
        '## Anchoring', '',
        f'- Explicit wrong-answer commitments: {len(anchored)} / {len(merged)}',
        f'- Anchored mean recovery: {anchored.recovery_rate.mean():.4f}',
        f'- Non-anchored mean recovery: {unanchored.recovery_rate.mean():.4f}', '',
        '## Within-Question Results', '',
        '| predictor | direction | valid questions | pairwise AUC | 95% CI | mean corr |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, row in summary.iterrows():
        lines.append(f'| `{row.feature}` | {row.direction} | {int(row.valid_pairwise_questions)} | {row.mean_within_pairwise_auc:.4f} | [{row.pairwise_auc_ci_low:.4f}, {row.pairwise_auc_ci_high:.4f}] | {row.mean_within_corr:.4f} |')
    lines += ['', '## Decision Rule', '', 'Do not collect a holdout or launch RL from a regex proxy. Use this table only to prioritize blinded human error-type annotation and a later frozen semantic predictor.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    scores = pd.read_csv(args.scores)
    for frame in (manifest, scores):
        frame['question_id'] = frame['question_id'].astype(str)
        frame['rollout_id'] = frame['rollout_id'].astype(int)
    merged = merge_manifest_scores(scores, manifest)
    merged = add_text_proxy_features(merged)
    predictor_directions = dict(PREDICTOR_DIRECTIONS)
    if args.annotations:
        annotations = pd.read_csv(args.annotations, keep_default_na=False)
        annotations['question_id'] = annotations['question_id'].astype(str)
        annotations['rollout_id'] = annotations['rollout_id'].astype(int)
        annotation_columns = ['question_id', 'rollout_id'] + [
            column for column in ('error_type', 'commitment_level', *YES_NO_FIELDS) if column in annotations
        ]
        merged = merged.merge(annotations[annotation_columns], on=['question_id', 'rollout_id'], how='left', validate='one_to_one')
        merged, annotation_directions = add_annotation_predictors(merged)
        predictor_directions.update(annotation_directions)
    summaries = []
    per_question = []
    for index, (predictor, direction) in enumerate(predictor_directions.items()):
        if predictor not in merged or merged[predictor].notna().sum() == 0:
            continue
        result, question_rows = analyze_predictor(merged, predictor, direction, args.bootstrap, args.seed + index * 17)
        summaries.append(result)
        per_question.append(question_rows)
    summary = pd.DataFrame(summaries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'error_type_proxy_rows.csv', index=False)
    summary.to_csv(output_dir / 'error_type_proxy_summary.csv', index=False)
    pd.concat(per_question, ignore_index=True).to_csv(output_dir / 'error_type_proxy_per_question.csv', index=False)
    _write_report(output_dir / 'ERROR_TYPE_PREDICTOR_DISCOVERY.md', merged, summary)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
