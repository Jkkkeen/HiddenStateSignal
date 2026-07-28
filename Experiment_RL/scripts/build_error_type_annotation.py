#!/usr/bin/env python3
'''Build an outcome-blinded annotation sheet for recoverability error types.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    'error_type',
    'error_subtype',
    'commitment_level',
    'reasoning_value_correct',
    'selected_option_mapping_correct',
    'visual_reread_likely_helpful',
    'annotation_confidence',
    'annotation_notes',
]


def build_blinded_annotation_rows(manifest: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest['question_id'] = manifest['question_id'].astype(str)
    scores = scores.copy()
    scores['question_id'] = scores['question_id'].astype(str)
    keys = ['question_id', 'rollout_id']
    expected = set(map(tuple, scores[keys].astype({'rollout_id': int}).to_numpy()))
    available = set(map(tuple, manifest[keys].astype({'rollout_id': int}).to_numpy()))
    missing = expected - available
    if missing:
        raise ValueError(f'{len(missing)} scored prefixes are missing from the manifest')

    rows = pd.DataFrame({
        'annotation_id': manifest['question_id'] + '_r' + manifest['rollout_id'].astype(str),
        'question_id': manifest['question_id'],
        'rollout_id': manifest['rollout_id'].astype(int),
        'prompt': manifest['prompt'],
        'image_path': manifest['image_path'],
        'gold_answer': manifest['answer'],
        'original_prediction': manifest['pred_answer'],
        'reasoning_prefix': manifest['revision_prefix'],
    })
    for column in ANNOTATION_COLUMNS:
        rows[column] = ''
    return rows.sort_values(['question_id', 'rollout_id']).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.DataFrame(
        [json.loads(line) for line in Path(args.manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    )
    scores = pd.read_csv(args.scores)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_blinded_annotation_rows(manifest, scores)
    rows.to_csv(output, index=False)
    print(f'Wrote {len(rows)} blinded rows to {output}')


if __name__ == '__main__':
    main()
