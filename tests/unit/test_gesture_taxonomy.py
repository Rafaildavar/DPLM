import json

import pytest

from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    labels_for_gesture_types,
    load_gesture_taxonomy,
    parse_gesture_type_scope,
)


def test_taxonomy_filters_dynamic_labels_from_config(tmp_path):
    taxonomy_path = tmp_path / "gesture_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "default_type": "static",
                "types": {
                    "static": ["palm"],
                    "quasi_static": ["hold_left"],
                    "dynamic": ["swipe_up"],
                },
                "patterns": {"dynamic": ["swipe_*"]},
            }
        ),
        encoding="utf-8",
    )

    labels = ["palm", "swipe_up", "swipe_down", "hold_left", "unknown"]

    assert labels_for_gesture_types(
        labels,
        GESTURE_TYPE_DYNAMIC,
        taxonomy_path=taxonomy_path,
    ) == ["swipe_up", "swipe_down"]
    assert labels_for_gesture_types(
        labels,
        [GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC],
        taxonomy_path=taxonomy_path,
    ) == ["palm", "hold_left", "unknown"]


def test_taxonomy_rejects_unknown_scope():
    with pytest.raises(ValueError):
        parse_gesture_type_scope("dynamic,magic")


def test_default_taxonomy_marks_swipes_dynamic():
    taxonomy = load_gesture_taxonomy()

    assert taxonomy.gesture_type_for_label("swipe_down") == GESTURE_TYPE_DYNAMIC
    assert taxonomy.gesture_type_for_label("SWIPE_RIGHT") == GESTURE_TYPE_DYNAMIC
