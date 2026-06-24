.PHONY: test-unit test-jmlc-profile test-jmlc-ml jmlc-profile jmlc-clean-profile compare-models

PYTHON ?= python

test-unit:
	$(PYTHON) -m pytest tests/unit -q

test-jmlc-profile:
	$(PYTHON) -m pytest tests/unit/test_jmlc_dataset_profile.py -q -o addopts=''

test-jmlc-ml:
	$(PYTHON) -m pytest \
		tests/unit/test_jmlc_dataset_profile.py \
		tests/unit/test_gesture_features.py \
		tests/unit/test_compare_models.py \
		-q -o addopts=''

jmlc-profile:
	$(PYTHON) -m scripts.jmlc_dataset_profile

jmlc-clean-profile:
	$(PYTHON) -m scripts.jmlc_dataset_profile \
		--json-out docs/experiments/dataset_profile.json \
		--markdown-out docs/experiments/dataset_profile.md

compare-models:
	$(PYTHON) -m scripts.compare_models
