from app.services.binding_agent import BindingAgentContext, BindingAgentOrchestrator
from app.services.binding_agents.task_frame import TaskFrameExtractor


GESTURES = [{"label": "palm"}, {"label": "zoom"}, {"label": "swipe_left"}]


def _frame(prompt: str):
    return TaskFrameExtractor().extract(BindingAgentContext(prompt, GESTURES))


def test_task_frame_prefers_binding_command_over_question_shape():
    frame = _frame("можешь ли ты открыть Safari жестом palm?")

    assert frame.intent == "create_binding"
    assert frame.operation.value == "create"
    assert frame.block == "binding"


def test_task_frame_models_delete_and_inspect_as_distinct_operations():
    delete = _frame("удали привязку для palm")
    inspect = _frame("покажи текущую привязку palm")

    assert delete.intent == "delete_binding"
    assert delete.operation.value == "delete"
    assert inspect.intent == "inspect_binding"
    assert inspect.operation.value == "inspect"


def test_task_frame_preserves_negation_instead_of_creating_binding():
    frame = _frame("не привязывай palm к открытию Safari")

    assert frame.intent == "cancel_binding"
    assert frame.negated is True
    assert frame.route == "answer"


def test_legacy_result_exposes_task_frame_and_explicit_status():
    result = BindingAgentOrchestrator().run(
        "привяжи zoom к повышению звука",
        GESTURES,
        provider="local",
    )
    draft = result.to_legacy_draft()

    assert draft["status"] == "ready"
    assert draft["taskFrame"]["operation"] == "create"
    assert draft["taskFrame"]["evidence"]


def test_clarification_result_has_explicit_status():
    result = BindingAgentOrchestrator().run(
        "жест swipe_left",
        GESTURES,
        provider="local",
    )

    assert str(result.status) == "clarify"
