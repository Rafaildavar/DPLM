import numpy as np

from cv.gesture_validation import make_grouped_splitter


def test_grouped_splitter_never_separates_source_family():
    y = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
    groups = np.asarray(
        ["a0", "a0", "a1", "a1", "a2", "a2", "b0", "b0", "b1", "b1", "b2", "b2"],
        dtype=object,
    )
    splitter, folds = make_grouped_splitter(
        y,
        groups,
        max_folds=3,
        random_state=7,
    )

    assert folds == 3
    for train_idx, validation_idx in splitter.split(
        np.zeros((len(y), 1), dtype=np.float32),
        y,
        groups=groups,
    ):
        assert set(groups[train_idx]).isdisjoint(set(groups[validation_idx]))
