.PHONY: ci test-unit test-unit-ci test-jmlc-profile test-jmlc-ml ml-smoke docker-build docker-ml-smoke docker-ci docker-mlflow macos-app jmlc-profile jmlc-clean-profile compare-models threshold-report negative-samples static-rejection-verifiers rejection-benchmark external-negative-experiments dynamic-prototype-experiments ipn-convert ipn-tar-convert ipn-external-analysis mlops-dashboard mlflow-ui

PYTHON ?= python
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
CI_ARTIFACT_DIR ?= outputs/ci
CI_PYTEST_TARGETS ?= tests/unit --ignore=tests/unit/test_app_controller.py

ci: test-unit-ci ml-smoke

test-unit-ci:
	$(PYTHON) -m pytest $(CI_PYTEST_TARGETS) --no-cov -q

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

ml-smoke:
	LOKY_MAX_CPU_COUNT=4 $(PYTHON) -m scripts.ml_smoke \
		--report-json $(CI_ARTIFACT_DIR)/ml_smoke_report.json

docker-build:
	docker build --platform linux/amd64 -t gestureflow-runtime:local .

docker-ml-smoke:
	docker compose --profile tools run --rm ml-smoke

docker-ci:
	docker compose --profile tools run --rm ci

docker-mlflow:
	docker compose --profile tools up mlflow

macos-app:
	bash packaging/macos/build_app.sh

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

static-rejection-verifiers:
	$(PYTHON) -m scripts.train_static_rejection_verifiers

rejection-benchmark:
	$(PYTHON) -m scripts.rejection_method_benchmark

external-negative-experiments:
	$(PYTHON) -m scripts.external_negative_dataset_experiments

dynamic-prototype-experiments:
	$(PYTHON) -m scripts.dynamic_prototype_experiments \
		--include-external-negatives \
		--write-production \
		--mlflow-tracking-uri $(MLFLOW_TRACKING_URI)

ipn-convert:
	$(PYTHON) -m scripts.convert_ipn_hand

ipn-tar-convert:
	$(PYTHON) -m scripts.convert_ipn_hand_tars \
		--limit-total 240 \
		--max-per-original-label 40 \
		--max-frames-per-segment 48 \
		--mlflow-tracking-uri $(MLFLOW_TRACKING_URI)

ipn-external-analysis:
	$(PYTHON) -m scripts.analyze_ipn_external_negatives

mlops-dashboard:
	$(PYTHON) -m scripts.mlops_dashboard

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --host 127.0.0.1 --port 5000
