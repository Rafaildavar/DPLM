.PHONY: test-unit test-jmlc-profile test-jmlc-ml jmlc-profile jmlc-clean-profile compare-models threshold-report negative-samples rejection-benchmark mlops-dashboard mlflow-ui

PYTHON ?= python
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

test-unit:
	$(PYTHON) -m pytest tests/unit -q

test-jmlc-profile:
	$(PYTHON) -m pytest tests/unit/test_jmlc_dataset_profile.py -q -o addopts=''

test-jmlc-ml:
	$(PYTHON) -m pytest \
		tests/unit/test_jmlc_dataset_profile.py \
		tests/unit/test_gesture_features.py \
		tests/unit/test_compare_models.py \
		tests/unit/test_threshold_report.py \
		-q -o addopts=''

jmlc-profile:
	$(PYTHON) -m scripts.jmlc_dataset_profile

jmlc-clean-profile:
	$(PYTHON) -m scripts.jmlc_dataset_profile \
		--json-out docs/experiments/dataset_profile.json \
		--markdown-out docs/experiments/dataset_profile.md

compare-models:
	$(PYTHON) -m scripts.compare_models

threshold-report:
	$(PYTHON) -m scripts.threshold_report

negative-samples:
	$(PYTHON) -m scripts.generate_negative_samples

rejection-benchmark:
	$(PYTHON) -m scripts.rejection_method_benchmark

mlops-dashboard:
	$(PYTHON) -m scripts.mlops_dashboard

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --host 127.0.0.1 --port 5000
