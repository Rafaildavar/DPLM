#!/usr/bin/env python3
"""Run the GestureBind golden set through the MAS and a 10-criterion judge."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.binding_agent import BindingAgentMlflowLogger, BindingAgentOrchestrator
from app.services.binding_agents.e2e_judge import (
    BindingAgentE2ECase,
    BindingAgentLlmJudge,
)
from app.services.binding_agents.eval_cases import BINDING_AGENT_EVAL_CASES


def _judge(mode: str) -> BindingAgentLlmJudge:
    if mode == "local":
        return BindingAgentLlmJudge()
    judge = BindingAgentLlmJudge.from_mistral_env()
    if mode == "mistral" and judge.completion_fn is None:
        raise RuntimeError("MISTRAL_API_KEY is required for --judge mistral")
    return judge


def _e2e_case(case) -> BindingAgentE2ECase:
    return BindingAgentE2ECase(
        case_id=case.case_id,
        title=case.notes or case.case_id.replace("_", " "),
        prompt=case.prompt,
        expected_intent=case.expected_intent,
        expected_block=case.expected_block,
        expected_mode=case.expected_mode,
        expected_can_apply=case.expected_can_apply,
        expected_gesture=case.expected_gesture,
        expected_action=case.expected_action,
        expected_action_spec=dict(case.expected_action_spec),
        expected_missing=case.expected_missing,
        expected_sequence_steps=case.expected_sequence_steps,
        current_gesture=case.current_gesture,
        gestures=case.gestures,
        notes=case.notes,
    )


def run_evaluation(judge_mode: str) -> dict[str, Any]:
    orchestrator = BindingAgentOrchestrator(
        mlflow_logger=BindingAgentMlflowLogger(enabled=False)
    )
    judge = _judge(judge_mode)
    rows: list[dict[str, Any]] = []
    criterion_values: dict[str, list[float]] = defaultdict(list)

    for golden in BINDING_AGENT_EVAL_CASES:
        result = orchestrator.run(
            golden.prompt,
            list(golden.gestures),
            current_gesture=golden.current_gesture,
            bindings=list(golden.bindings),
            provider="local",
        )
        report = judge.evaluate(_e2e_case(golden), result)
        report_data = report.to_dict()
        for score in report.scores:
            criterion_values[score.criterion_id].append(score.score)
        rows.append(
            {
                "case": golden.case_id,
                "tags": list(golden.tags),
                "passed": report.passed,
                "score": report.overall_score,
                "judgeSource": report.judge_source,
                "latencyMs": result.telemetry.get("latency_ms", 0.0),
                "result": {
                    "intent": result.intent,
                    "block": result.intent_block,
                    "mode": result.mode,
                    "canApply": result.can_apply,
                    "gesture": result.gesture_label,
                    "actionSpec": result.action_spec,
                    "missing": result.missing,
                    "mutation": result.mutation,
                },
                "judge": report_data,
            }
        )

    scores = [float(row["score"]) for row in rows]
    passed = sum(1 for row in rows if row["passed"])
    return {
        "suite": "gesturebind_binding_mas_golden",
        "cases": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "passRate": passed / max(1, len(rows)),
        "meanScore": mean(scores) if scores else 0.0,
        "criterionMeans": {
            criterion: mean(values)
            for criterion, values in sorted(criterion_values.items())
        },
        "rows": rows,
    }


def _log_mlflow(report: dict[str, Any]) -> None:
    import mlflow

    tracking_uri = str(os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000")
    experiment = str(
        os.getenv("DPLM_BINDING_AGENT_MLFLOW_EXPERIMENT")
        or "GestureBind-Evaluation"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="binding-mas-golden-evaluation"):
        mlflow.set_tags(
            {
                "run_kind": "binding_agent_evaluation",
                "suite": report["suite"],
            }
        )
        mlflow.log_metrics(
            {
                "cases": float(report["cases"]),
                "passed": float(report["passed"]),
                "failed": float(report["failed"]),
                "pass_rate": float(report["passRate"]),
                "mean_judge_score": float(report["meanScore"]),
                **{
                    f"judge_{name}": float(value)
                    for name, value in report["criterionMeans"].items()
                },
            }
        )
        mlflow.log_dict(report, "binding_mas_evaluation.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=("local", "auto", "mistral"), default="local")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_evaluation(args.judge)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.mlflow:
        _log_mlflow(report)

    print(
        f"GestureBind MAS: {report['passed']}/{report['cases']} passed, "
        f"mean judge score={report['meanScore']:.3f}"
    )
    for row in report["rows"]:
        if not row["passed"]:
            print(f"FAIL {row['case']}: score={row['score']:.3f}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
