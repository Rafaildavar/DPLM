"""Lightweight semantic intent routing for the binding MAS.

The router intentionally has no network/model dependency. It behaves like a
small local embedding layer: prompts and intent examples are converted into a
sparse feature vector, then compared with cosine similarity.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class SemanticRoute:
    intent: str
    block: str
    route: str
    score: float
    matched_example: str
    margin: float = 0.0
    method: str = "hybrid_embedding"
    alternatives: tuple[tuple[str, float, str], ...] = ()


INTENT_ROUTE_TABLE: dict[str, tuple[str, str]] = {
    "create_binding": ("binding", "binding_pipeline"),
    "update_binding": ("binding", "binding_pipeline"),
    "build_sequence": ("binding", "binding_pipeline"),
    "validate_command": ("project_question", "answer"),
    "project_question": ("project_question", "answer"),
    "unsupported_general_question": ("unsupported_general", "safe_redirect"),
}


INTENT_EXAMPLES: dict[str, tuple[str, ...]] = {
    "create_binding": (
        "привяжи жест к открытию приложения",
        "назначь ладонь на запуск safari",
        "сохрани свайп вверх как command z",
        "gesture palm open browser",
        "хочу чтобы этот жест нажимал hotkey",
    ),
    "update_binding": (
        "измени привязку жеста на другую команду",
        "поменяй действие у свайпа вверх",
        "замени команду для жеста",
        "переназначь текущий жест",
        "обнови binding на новый hotkey",
    ),
    "build_sequence": (
        "собери сценарий из нескольких шагов",
        "сделай утренний ритуал открыть почту jira заметки",
        "рабочий старт почта таски календарь chatgpt",
        "создай routine чтобы последовательно открыть приложения",
        "первый шаг открыть почту второе открыть задачи потом заметки",
    ),
    "validate_command": (
        "разве command z закрывает приложение",
        "правильно ли hotkey выполняет это действие",
        "подходит ли сочетание клавиш для закрытия telegram",
        "будет ли command q завершать приложение",
        "соответствует ли команда ожидаемому действию",
    ),
    "project_question": (
        "что умеет агент привязки gesturebind",
        "какие жесты и команды доступны",
        "как работает mlflow трассировка агентов",
        "как устроена система привязок",
        "почему агент остается в рамках проекта",
    ),
    "unsupported_general_question": (
        "какая завтра погода",
        "расскажи новости",
        "какой курс валют",
        "посоветуй фильм",
        "помоги с домашним заданием не по проекту",
    ),
}

SEMANTIC_ROUTE_THRESHOLD = 0.27
SEMANTIC_SEQUENCE_OVERRIDE_THRESHOLD = 0.24
SEMANTIC_MIN_MARGIN = 0.045


def _normalize(text: str) -> str:
    value = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9]+", _normalize(text), re.IGNORECASE)


def _stem(token: str) -> str:
    if len(token) <= 5:
        return token
    for suffix in (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ыми",
        "ими",
        "ить",
        "ать",
        "ешь",
        "ает",
        "ают",
        "ого",
        "ий",
        "ый",
        "ая",
        "ое",
        "ые",
        "ов",
        "ев",
        "ом",
        "ем",
        "ам",
        "ям",
        "ах",
        "ях",
        "и",
        "ы",
        "а",
        "я",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _features(text: str) -> Counter[str]:
    tokens = _tokens(text)
    features: Counter[str] = Counter()
    for token in tokens:
        stem = _stem(token)
        features[f"w:{token}"] += 2
        features[f"s:{stem}"] += 2
        if len(token) >= 5:
            features[f"p:{token[:5]}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[f"b:{_stem(left)} {_stem(right)}"] += 3
    joined = " ".join(tokens)
    for index in range(max(0, len(joined) - 2)):
        gram = joined[index : index + 3]
        if " " not in gram:
            features[f"c:{gram}"] += 1
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[item] * right[item] for item in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


@lru_cache(maxsize=1)
def _embedding_index():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return None
    rows: list[tuple[str, str]] = []
    for intent, examples in INTENT_EXAMPLES.items():
        rows.extend((intent, example) for example in examples)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([example for _intent, example in rows])
    return vectorizer, matrix, tuple(rows)


def _embedding_scores(text: str) -> dict[tuple[str, str], float]:
    index = _embedding_index()
    if index is None:
        return {}
    vectorizer, matrix, rows = index
    query = vectorizer.transform([_normalize(text)])
    similarities = (matrix @ query.T).toarray().ravel()
    return {
        (intent, example): float(score)
        for (intent, example), score in zip(rows, similarities)
    }


def route_semantically(text: str) -> SemanticRoute:
    query = _features(text)
    embedding_scores = _embedding_scores(text)
    by_intent: dict[str, tuple[float, str]] = {}
    for intent, examples in INTENT_EXAMPLES.items():
        for example in examples:
            lexical = _cosine(query, _features(example))
            embedding = embedding_scores.get((intent, example), 0.0)
            score = (0.48 * lexical) + (0.52 * embedding)
            current = by_intent.get(intent)
            if current is None or score > current[0]:
                by_intent[intent] = (score, example)
    ranked = sorted(
        (
            (intent, score, example)
            for intent, (score, example) in by_intent.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_intent, best_score, best_example = (
        ranked[0]
        if ranked
        else ("unsupported_general_question", 0.0, "")
    )
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    block, route = INTENT_ROUTE_TABLE[best_intent]
    return SemanticRoute(
        intent=best_intent,
        block=block,
        route=route,
        score=best_score,
        matched_example=best_example,
        margin=max(0.0, best_score - second_score),
        alternatives=tuple(ranked[:3]),
    )
