"""Executable skill registry for safe action candidate preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.services.binding_agents.contracts import ActionCandidate, RiskLevel, TaskFrame
from app.services.binding_agents.skills.base import AgentSkill


@dataclass(frozen=True)
class ExecutableSkill:
    manifest: AgentSkill

    def can_handle(self, action_spec: dict[str, Any], frame: TaskFrame | None) -> bool:
        action = str(action_spec.get("action") or "")
        if action not in self.manifest.actions:
            return False
        if frame is not None and self.manifest.intents:
            return frame.intent in self.manifest.intents
        return True

    def validate(self, action_spec: dict[str, Any]) -> tuple[str, ...]:
        from app.services.user_command_sync import validate_action_spec

        error = validate_action_spec(action_spec)
        return (error,) if error else ()


class SkillRuntime:
    def __init__(self, skills: Iterable[ExecutableSkill]) -> None:
        self._skills = tuple(skills)
        self._by_id = {skill.manifest.skill_id: skill for skill in self._skills}

    @property
    def manifests(self) -> tuple[AgentSkill, ...]:
        return tuple(skill.manifest for skill in self._skills)

    def select(
        self,
        action_spec: dict[str, Any],
        *,
        frame: TaskFrame | None = None,
        preferred_skill: str = "",
    ) -> ExecutableSkill | None:
        preferred = self._by_id.get(preferred_skill)
        if preferred and preferred.can_handle(action_spec, frame):
            return preferred
        return next(
            (skill for skill in self._skills if skill.can_handle(action_spec, frame)),
            None,
        )

    def prepare_candidate(
        self,
        action_spec: dict[str, Any],
        *,
        frame: TaskFrame | None,
        preferred_skill: str,
        source: str,
        confidence: float,
        evidence: tuple[str, ...],
        risk: RiskLevel,
        requires_confirmation: bool,
    ) -> ActionCandidate:
        skill = self.select(
            action_spec,
            frame=frame,
            preferred_skill=preferred_skill,
        )
        if skill is None:
            return ActionCandidate(
                action_spec,
                source,
                confidence,
                evidence=evidence,
                issues=("no_executable_skill",),
            )
        issues = skill.validate(action_spec)
        manifest_risk = RiskLevel(skill.manifest.risk)
        selected_risk = (
            manifest_risk
            if _risk_rank(manifest_risk) >= _risk_rank(risk)
            else risk
        )
        return ActionCandidate(
            action_spec=action_spec,
            source=source,
            confidence=confidence,
            evidence=evidence,
            issues=issues,
            skill_id=skill.manifest.skill_id,
            skill_version=skill.manifest.version,
            risk=selected_risk,
            requires_confirmation=(
                requires_confirmation or selected_risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}
            ),
        )


def _risk_rank(value: RiskLevel) -> int:
    return {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}[value]


def _skill(
    skill_id: str,
    title: str,
    description: str,
    actions: tuple[str, ...],
    examples: tuple[str, ...],
    *,
    negative: tuple[str, ...] = (),
    risk: str = "low",
    permissions: tuple[str, ...] = (),
) -> ExecutableSkill:
    return ExecutableSkill(
        AgentSkill(
            skill_id=skill_id,
            title=title,
            description=description,
            intents=("create_binding", "update_binding", "build_sequence"),
            actions=actions,
            positive_examples=examples,
            negative_examples=negative,
            input_schema="TaskFrame + action parameters",
            output_schema="ActionCandidate[actionSpec]",
            required_permissions=permissions,
            risk=risk,
        )
    )


DEFAULT_EXECUTABLE_SKILLS: tuple[ExecutableSkill, ...] = (
    _skill(
        "macos.app.open",
        "Open macOS App",
        "Открывает явно названное приложение macOS.",
        ("open_app",),
        ("открыть Safari", "запустить Telegram"),
        negative=("закрыть Telegram", "включить Bluetooth"),
    ),
    _skill(
        "macos.app.quit",
        "Quit macOS App",
        "Завершает явно названное приложение после подтверждения.",
        ("quit_app",),
        ("закрыть Telegram", "quit Safari"),
        negative=("открыть Telegram",),
        risk="medium",
        permissions=("automation",),
    ),
    _skill(
        "browser.open_url",
        "Open URL",
        "Открывает проверенный http/https URL.",
        ("open_url",),
        ("открыть https://example.com",),
        negative=("открыть неизвестный сайт без URL",),
    ),
    _skill(
        "macos.path.open",
        "Open Path",
        "Открывает существующий файл или каталог.",
        ("open_path",),
        ("открыть /Users/me/Documents",),
    ),
    _skill(
        "macos.keyboard.hotkey",
        "Keyboard Shortcut",
        "Отправляет проверенное сочетание клавиш.",
        ("key_combination",),
        ("command+z", "ctrl+left"),
        permissions=("accessibility",),
    ),
    _skill(
        "macos.spaces.navigate",
        "Navigate Spaces",
        "Переключает рабочие столы macOS.",
        ("key_combination",),
        ("перейти на рабочий стол слева",),
        permissions=("accessibility",),
    ),
    _skill(
        "macos.page.zoom",
        "Page Zoom",
        "Меняет масштаб активной страницы сочетанием клавиш.",
        ("key_combination",),
        ("увеличить масштаб страницы", "уменьшить zoom"),
        permissions=("accessibility",),
    ),
    _skill(
        "macos.keyboard.press",
        "Press Key",
        "Нажимает одну нормализованную клавишу.",
        ("press",),
        ("нажать пробел", "press escape"),
        permissions=("accessibility",),
    ),
    _skill(
        "macos.page.scroll",
        "Scroll Page",
        "Прокручивает активную область.",
        ("scroll",),
        ("прокрутить вниз",),
        permissions=("accessibility",),
    ),
    _skill(
        "macos.media.control",
        "Media Control",
        "Управляет воспроизведением медиа.",
        ("media_key",),
        ("поставить видео на паузу", "следующий трек"),
    ),
    _skill(
        "macos.volume.control",
        "Volume Control",
        "Изменяет громкость или mute.",
        ("volume_up", "volume_down", "mute_toggle"),
        ("сделать громче", "убавить звук"),
    ),
    _skill(
        "macos.brightness.control",
        "Brightness Control",
        "Изменяет яркость экрана.",
        ("brightness_up", "brightness_down"),
        ("сделать экран ярче",),
    ),
    _skill(
        "macos.screen.lock",
        "Lock Screen",
        "Блокирует экран после явного пользовательского действия.",
        ("lock_screen",),
        ("заблокировать экран",),
        risk="high",
    ),
    _skill(
        "macos.screen.capture",
        "Capture Screen",
        "Создаёт снимок экрана.",
        ("screenshot",),
        ("сделать скриншот",),
        permissions=("screen_recording",),
    ),
    _skill(
        "macos.notification.show",
        "Show Notification",
        "Показывает локальное уведомление.",
        ("notify",),
        ("показать уведомление готово",),
    ),
    _skill(
        "scenario.wait",
        "Wait",
        "Добавляет ограниченную паузу в сценарий.",
        ("wait",),
        ("подождать 2 секунды",),
    ),
    _skill(
        "macos.script.run",
        "Run Approved Script",
        "Запускает существующий Python-скрипт после проверки и подтверждения.",
        ("run_script",),
        ("запустить /path/tool.py",),
        risk="high",
    ),
    _skill(
        "scenario.compose",
        "Compose Scenario",
        "Собирает последовательность проверенных action skills.",
        ("sequence",),
        ("открыть Safari, потом показать уведомление",),
    ),
)


DEFAULT_SKILL_RUNTIME = SkillRuntime(DEFAULT_EXECUTABLE_SKILLS)


__all__ = [
    "DEFAULT_EXECUTABLE_SKILLS",
    "DEFAULT_SKILL_RUNTIME",
    "ExecutableSkill",
    "SkillRuntime",
]
