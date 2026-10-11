"""Tests for the conservative object -> material mapping."""

from __future__ import annotations

from src.vision.material_mapping import (
    GLASS,
    ORGANIC,
    PAPER,
    PLASTIC,
    hazard_note_for_label,
    map_label_to_material,
)


def test_known_labels_map():
    assert map_label_to_material("water bottle").category_slug == PLASTIC
    assert map_label_to_material("wine bottle").category_slug == GLASS
    assert map_label_to_material("carton").category_slug == PAPER
    assert map_label_to_material("banana").category_slug == ORGANIC


def test_case_insensitive():
    assert map_label_to_material("WATER BOTTLE").category_slug == PLASTIC


def test_unknown_label_returns_none():
    assert map_label_to_material("tench") is None
    assert map_label_to_material("") is None


def test_hazard_note_present_for_battery_bearing_objects():
    assert hazard_note_for_label("iPod") is not None
    assert hazard_note_for_label("banana") is None


def test_mapping_is_not_claimed_as_direct_classifier():
    hint = map_label_to_material("carton")
    assert hint.is_direct_material_classifier is False
