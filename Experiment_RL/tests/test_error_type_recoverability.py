from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Experiment_RL' / 'scripts'))

from analyze_error_type_predictors import add_annotation_predictors, add_text_proxy_features, analyze_predictor, merge_manifest_scores
from build_error_type_annotation import build_blinded_annotation_rows


def test_blinded_annotation_rows_exclude_outcomes_and_margin_signals() -> None:
    manifest = pd.DataFrame([{'question_id': 'q1', 'rollout_id': 2, 'prompt': 'Question', 'image_path': 'image.png', 'revision_prefix': 'Reasoning', 'answer': 'C', 'pred_answer': 'A', 'trimmed_margin_mean': 3.0}])
    scores = pd.DataFrame([{'question_id': 'q1', 'rollout_id': 2, 'recovery_rate': 1.0}])
    rows = build_blinded_annotation_rows(manifest, scores)
    assert rows.loc[0, 'annotation_id'] == 'q1_r2'
    assert rows.loc[0, 'gold_answer'] == 'C'
    assert rows.loc[0, 'original_prediction'] == 'A'
    assert 'recovery_rate' not in rows.columns
    assert 'trimmed_margin_mean' not in rows.columns


def test_text_proxy_features_detect_expected_markers() -> None:
    frame = pd.DataFrame([{'revision_prefix': 'The diagram shows 12. Wait, reread the image. This corresponds to option B. Final answer is B.', 'original_prediction_claim_flag': True}])
    enriched = add_text_proxy_features(frame)
    assert enriched.loc[0, 'explicit_wrong_commitment'] == 1.0
    assert enriched.loc[0, 'self_correction_marker_count'] >= 1
    assert enriched.loc[0, 'visual_reference_count'] >= 2
    assert enriched.loc[0, 'option_mapping_count'] >= 2
    assert enriched.loc[0, 'self_correction_per_1k_tokens'] > 0


def test_analyze_predictor_uses_within_question_ordering() -> None:
    rows = []
    for question_id in ('q1', 'q2'):
        for rollout_id, outcome in enumerate((0.0, 0.25, 0.75, 1.0)):
            rows.append({'question_id': question_id, 'rollout_id': rollout_id, 'recovery_rate': outcome, 'proxy': float(rollout_id)})
    result, per_question = analyze_predictor(pd.DataFrame(rows), predictor='proxy', direction='pos', n_boot=100, seed=7)
    assert result['mean_within_pairwise_auc'] == 1.0
    assert result['valid_pairwise_questions'] == 2
    assert len(per_question) == 2


def test_merge_manifest_scores_preserves_existing_commitment_flag() -> None:
    scores = pd.DataFrame([{'question_id': 'q1', 'rollout_id': 1, 'recovery_rate': 0.5, 'original_prediction_claim_flag': True}])
    manifest = pd.DataFrame([{'question_id': 'q1', 'rollout_id': 1, 'revision_prefix': 'reasoning', 'original_prediction_claim_flag': True}])
    merged = merge_manifest_scores(scores, manifest)
    assert merged.loc[0, 'original_prediction_claim_flag']
    assert 'original_prediction_claim_flag_x' not in merged.columns
    assert 'original_prediction_claim_flag_y' not in merged.columns


def test_annotation_predictors_encode_semantic_labels() -> None:
    frame = pd.DataFrame([{'error_type': 'option_mapping', 'commitment_level': '2', 'reasoning_value_correct': 'yes', 'visual_reread_likely_helpful': 'no'}])
    encoded, directions = add_annotation_predictors(frame)
    assert encoded.loc[0, 'error_type__option_mapping'] == 1.0
    assert encoded.loc[0, 'commitment_level'] == 2.0
    assert encoded.loc[0, 'reasoning_value_correct__yes'] == 1.0
    assert encoded.loc[0, 'visual_reread_likely_helpful__yes'] == 0.0
    assert directions['error_type__option_mapping'] == 'pos'
    assert directions['commitment_level'] == 'neg'
