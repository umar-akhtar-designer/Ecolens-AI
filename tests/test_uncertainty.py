"""Tests for uncertainty summarisation."""

from __future__ import annotations

import numpy as np

from src.vision import uncertainty as u


def test_confident_distribution():
    probs = np.array([0.9, 0.05, 0.03, 0.02])
    report = u.analyse(probs, num_classes=4)
    assert report.status == u.CONFIDENT
    assert report.margin > 0.5


def test_flat_distribution_is_low_or_uncertain():
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    report = u.analyse(probs, num_classes=4)
    assert report.status in (u.UNCERTAIN, u.LOW_CONFIDENCE)


def test_low_top_score_is_low_confidence():
    probs = np.array([0.1, 0.09, 0.08, 0.07])
    report = u.analyse(probs, num_classes=4)
    assert report.status == u.LOW_CONFIDENCE


def test_score_type_is_labelled_uncalibrated():
    assert "uncalibrated" in u.SCORE_TYPE


def test_entropy_bounds():
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    report = u.analyse(probs, num_classes=4)
    assert 0.0 <= report.normalized_entropy <= 1.0


def test_low_confidence_abstains():
    report = u.analyse(np.array([0.1, 0.09, 0.08, 0.07]), num_classes=4)
    assert report.abstain is True
    assert report.abstain_reason


def test_confident_does_not_abstain():
    report = u.analyse(np.array([0.9, 0.05, 0.03, 0.02]), num_classes=4)
    assert report.abstain is False
    assert report.supported_domain is True


def test_unsupported_domain_abstains_even_if_confident():
    report = u.analyse(np.array([0.9, 0.05, 0.03, 0.02]), num_classes=4, supported_domain=False)
    assert report.supported_domain is False
    assert report.abstain is True
    assert report.status != u.CONFIDENT


def test_status_labels_cover_states():
    for status in (u.CONFIDENT, u.UNCERTAIN, u.LOW_CONFIDENCE, u.UNKNOWN):
        assert status in u.STATUS_LABELS
