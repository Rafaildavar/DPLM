from __future__ import annotations

from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONT_PATH = Path("/Library/Fonts/Arial Unicode.ttf")

W, H = 1800, 1050
BG = "#fbfbfd"
INK = "#17202a"
MUTED = "#5f6b7a"
BLUE = "#2f67d8"
BLUE_LIGHT = "#dbe8ff"
GREEN = "#218c61"
GREEN_LIGHT = "#dff4e9"
ORANGE = "#b86e00"
ORANGE_LIGHT = "#fff0d8"
RED = "#b54747"
RED_LIGHT = "#fde2e2"
PURPLE = "#7a4cc2"
PURPLE_LIGHT = "#ece2ff"
GRAY = "#e7eaf0"
LINE = "#8a95a5"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Arial Unicode has Cyrillic glyphs and is available on the target macOS.
    return ImageFont.truetype(str(FONT_PATH), size=size)


F_TITLE = font(46)
F_SUB = font(28)
F_BOX = font(30)
F_SMALL = font(24)
F_TINY = font(19)


def canvas(title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 130), fill="#ffffff")
    d.line((0, 130, W, 130), fill=GRAY, width=3)
    display_title = title.split(" - ", 1)[1] if title.startswith("Рисунок ") and " - " in title else title
    title_font = F_TITLE if len(display_title) <= 64 else font(38)
    title_text = wrap(display_title, 72)
    d.multiline_text((70, 28), title_text, fill=INK, font=title_font, spacing=4)
    if subtitle:
        d.text((70, 98), subtitle, fill=MUTED, font=F_TINY)
    return img, d


def wrap(text: str, chars: int) -> str:
    lines = []
    for part in str(text).split("\n"):
        lines.extend(textwrap.wrap(part, width=chars) or [""])
    return "\n".join(lines)


def centered_text(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: str = INK,
    f: ImageFont.FreeTypeFont = F_BOX,
    chars: int = 18,
) -> None:
    x1, y1, x2, y2 = box
    txt = wrap(text, chars)
    bbox = d.multiline_textbbox((0, 0), txt, font=f, spacing=8, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.multiline_text(
        (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2),
        txt,
        fill=fill,
        font=f,
        spacing=8,
        align="center",
    )


def box(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str = BLUE_LIGHT,
    outline: str = BLUE,
    chars: int = 17,
    f: ImageFont.FreeTypeFont = F_BOX,
) -> None:
    d.rounded_rectangle(xy, radius=24, fill=fill, outline=outline, width=4)
    centered_text(d, xy, text, f=f, chars=chars)


def small_box(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str = "#ffffff",
    outline: str = LINE,
    chars: int = 20,
) -> None:
    d.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    centered_text(d, xy, text, f=F_SMALL, chars=chars)


def arrow(
    d: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = LINE,
    width: int = 5,
) -> None:
    d.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 18
    pts = [
        end,
        (
            int(end[0] - size * math.cos(angle - math.pi / 6)),
            int(end[1] - size * math.sin(angle - math.pi / 6)),
        ),
        (
            int(end[0] - size * math.cos(angle + math.pi / 6)),
            int(end[1] - size * math.sin(angle + math.pi / 6)),
        ),
    ]
    d.polygon(pts, fill=color)


def poly_arrow(
    d: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: str = LINE,
    width: int = 5,
) -> None:
    if len(points) < 2:
        return
    for a, b in zip(points, points[1:-1]):
        d.line((a, b), fill=color, width=width)
    arrow(d, points[-2], points[-1], color=color, width=width)


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, quality=95)


def fig_1_1() -> None:
    img, d = canvas(
        "Рисунок 1.1 - Схема представления руки через 21 ключевую точку MediaPipe Hands",
        "21 landmark: запястье, суставы и кончики пальцев",
    )
    pts = {
        0: (850, 830),
        1: (740, 710),
        2: (650, 600),
        3: (570, 510),
        4: (500, 430),
        5: (820, 630),
        6: (780, 480),
        7: (760, 360),
        8: (745, 250),
        9: (920, 620),
        10: (930, 455),
        11: (935, 320),
        12: (940, 205),
        13: (1040, 650),
        14: (1090, 505),
        15: (1125, 390),
        16: (1150, 290),
        17: (1160, 705),
        18: (1240, 595),
        19: (1300, 505),
        20: (1350, 425),
    }
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
        [5, 9, 13, 17],
    ]
    palm = [(720, 690), (850, 590), (1040, 610), (1190, 700), (1040, 850), (840, 850)]
    d.polygon(palm, fill="#fff4dc", outline="#d19a45")
    for seq in fingers:
        for a, b in zip(seq, seq[1:]):
            d.line((pts[a], pts[b]), fill=BLUE, width=8)
    for idx, p in pts.items():
        d.ellipse((p[0] - 15, p[1] - 15, p[0] + 15, p[1] + 15), fill="#ffffff", outline=BLUE, width=5)
        d.text((p[0] + 18, p[1] - 18), str(idx), fill=INK, font=F_TINY)
    small_box(d, (90, 220, 510, 340), "Вход: кадр RGB\nс веб-камеры", fill=BLUE_LIGHT, outline=BLUE)
    small_box(d, (90, 430, 510, 570), "MediaPipe Hand\nLandmarker", fill=GREEN_LIGHT, outline=GREEN)
    small_box(d, (90, 660, 510, 820), "Выход: координаты\n21 ключевой точки", fill=ORANGE_LIGHT, outline=ORANGE)
    arrow(d, (300, 340), (300, 430))
    arrow(d, (300, 570), (300, 660))
    save(img, "fig_1_1_mediapipe_landmarks.png")


def horizontal_flow(
    name: str,
    title: str,
    steps: list[str],
    subtitle: str | None = None,
    colors: list[tuple[str, str]] | None = None,
) -> None:
    img, d = canvas(title, subtitle)
    n = len(steps)
    margin = 90
    gap = 38
    y = 430
    bw = int((W - 2 * margin - gap * (n - 1)) / n)
    bh = 170
    colors = colors or [(BLUE_LIGHT, BLUE)] * n
    for i, step in enumerate(steps):
        x1 = margin + i * (bw + gap)
        xy = (x1, y, x1 + bw, y + bh)
        fill, outline = colors[i % len(colors)]
        box(d, xy, step, fill=fill, outline=outline, chars=13 if n > 6 else 16, f=F_SMALL if n > 6 else F_BOX)
        if i < n - 1:
            arrow(d, (x1 + bw + 8, y + bh // 2), (x1 + bw + gap - 8, y + bh // 2))
    save(img, name)


def fig_2_1() -> None:
    horizontal_flow(
        "fig_2_1_user_scenario.png",
        "Рисунок 2.1 - Основной пользовательский сценарий GestureBind",
        [
            "Запуск\nприложения",
            "Проверка\nсостояния",
            "Запись\nжеста",
            "Обучение\nмодели",
            "Привязка\nкоманды",
            "Распознавание",
            "Выполнение\nдействия",
            "Журнал\nсобытий",
        ],
        "Полный цикл: от пользовательского словаря до выполнения команды",
        [(BLUE_LIGHT, BLUE), (GREEN_LIGHT, GREEN), (ORANGE_LIGHT, ORANGE), (PURPLE_LIGHT, PURPLE)],
    )


def fig_2_2() -> None:
    img, d = canvas("Рисунок 2.2 - Общая архитектура GestureBind", "Слоистая архитектура: GUI отделен от CV/ML, ORM и команд")
    layers = [
        ((120, 175, 1680, 310), "GUI: Flet views\nhome, training, gestures,\nbindings, settings", BLUE_LIGHT, BLUE),
        ((120, 365, 1680, 500), "AppController\nкамера, события, состояние,\nзапуск процессов", GREEN_LIGHT, GREEN),
        ((120, 600, 510, 770), "CV-слой\nOpenCV + MediaPipe\nlandmarks", ORANGE_LIGHT, ORANGE),
        ((555, 600, 945, 770), "ML-слой\nKNN, classes.json,\nfeature_dim", PURPLE_LIGHT, PURPLE),
        ((990, 600, 1290, 770), "ORM/БД\nSQLAlchemy\nPostgreSQL/SQLite", "#eef2f5", "#596275"),
        ((1335, 600, 1680, 770), "Команды\npolicy + executor\nmacOS actions", RED_LIGHT, RED),
        ((360, 850, 760, 980), "data/gestures\nsample_*.npy", "#ffffff", LINE),
        ((860, 850, 1220, 980), "models\nknn.pkl, classes.json", "#ffffff", LINE),
    ]
    for i, (xy, text, fill, outline) in enumerate(layers):
        box(d, xy, text, fill=fill, outline=outline, chars=24, f=F_TINY if i < 2 else F_SMALL)
    arrow(d, (900, 310), (900, 365))
    poly_arrow(d, [(880, 515), (880, 565), (320, 565), (320, 600)])
    poly_arrow(d, [(900, 515), (900, 565), (750, 565), (750, 600)])
    poly_arrow(d, [(920, 515), (920, 565), (1140, 565), (1140, 600)])
    poly_arrow(d, [(940, 515), (940, 565), (1500, 565), (1500, 600)])
    arrow(d, (520, 850), (560, 770))
    arrow(d, (1040, 850), (820, 770))
    save(img, "fig_2_2_architecture.png")


def fig_2_3() -> None:
    img, d = canvas("Рисунок 2.3 - Логическая схема хранения данных GestureBind", "Основные ORM-сущности и связи пользовательского словаря")
    boxes = {
        "users": (80, 210, 390, 370),
        "gestures": (520, 190, 900, 410),
        "gesture_samples": (1030, 190, 1450, 410),
        "commands": (520, 520, 900, 720),
        "recognition_models": (1030, 520, 1450, 720),
        "recognition_logs": (80, 560, 390, 760),
        "settings": (1490, 560, 1730, 720),
        "gesture_history": (520, 820, 900, 960),
        "app_sessions": (80, 830, 390, 960),
    }
    details = {
        "users": "users\nid, username",
        "gestures": "gestures\nlabel, samples_path,\nmodel_class_id,\nis_active",
        "gesture_samples": "gesture_samples\ngesture_id,\nsample_index,\nfeatures_path",
        "commands": "commands\ngesture_id,\naction_type,\naction_spec",
        "recognition_models": "recognition_models\nmodel_path,\nclasses_path,\nfeature_dim",
        "recognition_logs": "recognition_logs\nlabel, confidence,\nexecuted",
        "settings": "settings\nthreshold,\ncooldown",
        "gesture_history": "gesture_history\ngesture_id,\ncommand_id",
        "app_sessions": "app_sessions\nstarted_at,\nended_at",
    }
    for name, xy in boxes.items():
        fill = GREEN_LIGHT if name in {"gestures", "gesture_samples"} else BLUE_LIGHT if name in {"commands", "gesture_history"} else "#ffffff"
        outline = GREEN if name in {"gestures", "gesture_samples"} else BLUE if name in {"commands", "gesture_history"} else LINE
        box(d, xy, details[name], fill=fill, outline=outline, chars=20, f=F_SMALL)
    def c(name, side):
        x1, y1, x2, y2 = boxes[name]
        return {
            "r": (x2, (y1 + y2) // 2),
            "l": (x1, (y1 + y2) // 2),
            "t": ((x1 + x2) // 2, y1),
            "b": ((x1 + x2) // 2, y2),
        }[side]
    arrow(d, c("users", "r"), c("gestures", "l"))
    arrow(d, c("gestures", "r"), c("gesture_samples", "l"))
    arrow(d, c("gestures", "b"), c("commands", "t"))
    arrow(d, c("commands", "b"), c("gesture_history", "t"))
    arrow(d, c("app_sessions", "t"), c("recognition_logs", "b"))
    poly_arrow(d, [c("gestures", "l"), (430, 300), (430, 660), c("recognition_logs", "r")])
    poly_arrow(d, [c("recognition_models", "l"), (960, 620), (960, 780), (430, 780), c("recognition_logs", "r")])
    save(img, "fig_2_3_database_schema.png")


def fig_2_4() -> None:
    horizontal_flow(
        "fig_2_4_main_screens.png",
        "Рисунок 2.4 - Основные экраны интерфейса GestureBind",
        ["Главная\nраспознавание", "Обучение\nсэмплы", "Жесты\nсловарь", "Привязки\nкоманды", "Настройки\nдиагностика"],
        "Постоянная навигация ведет пользователя по основным рабочим сценариям",
        [(BLUE_LIGHT, BLUE), (GREEN_LIGHT, GREEN), (ORANGE_LIGHT, ORANGE), (PURPLE_LIGHT, PURPLE), (RED_LIGHT, RED)],
    )


def fig_3_1() -> None:
    img, d = canvas("Рисунок 3.1 - Структура программных модулей GestureBind", "Фактические каталоги проекта")
    cols = [
        ("app/flet_app", "GUI\nviews + controller", BLUE_LIGHT, BLUE),
        ("app/services", "Сервисы\nкоманды, настройки,\nдиагностика, ORM sync", GREEN_LIGHT, GREEN),
        ("cv", "Компьютерное зрение\nrecord, train,\ninfer, landmarker", ORANGE_LIGHT, ORANGE),
        ("models", "Артефакты модели\nknn.pkl\nclasses.json", PURPLE_LIGHT, PURPLE),
        ("data/gestures", "Датасет\nsample_*.npy", "#ffffff", LINE),
        ("tests", "Проверки\nunit + integration", RED_LIGHT, RED),
    ]
    x = 95
    for i, (head, body, fill, outline) in enumerate(cols):
        y = 190 + (i % 3) * 235
        x = 95 + (i // 3) * 860
        box(d, (x, y, x + 690, y + 160), head + "\n" + body, fill=fill, outline=outline, chars=28, f=F_SMALL)
    d.line((880, 175, 880, 935), fill=GRAY, width=4)
    save(img, "fig_3_1_module_structure.png")


def fig_3_2() -> None:
    img, d = canvas("Рисунок 3.2 - Формирование признакового вектора из landmarks кисти", "Текущий локальный набор использует одну руку: 21 точка x 2 координаты = 42 признака")
    steps = [
        ("Кадры\nT=30", BLUE_LIGHT, BLUE),
        ("Landmarks\n(T, 21, 2)", GREEN_LIGHT, GREEN),
        ("Нормализация\nотносительно руки", ORANGE_LIGHT, ORANGE),
        ("Агрегация\nпо времени", PURPLE_LIGHT, PURPLE),
        ("Вектор\n42 признака", RED_LIGHT, RED),
    ]
    x0, y, bw, bh, gap = 105, 410, 260, 170, 70
    for i, (text, fill, outline) in enumerate(steps):
        x = x0 + i * (bw + gap)
        box(d, (x, y, x + bw, y + bh), text, fill=fill, outline=outline, chars=16, f=F_SMALL)
        if i < len(steps) - 1:
            arrow(d, (x + bw + 10, y + bh // 2), (x + bw + gap - 10, y + bh // 2))
    d.text((125, 720), "Для двух рук размерность может увеличиваться до 84 признаков: 2 x 21 x 2.", fill=MUTED, font=F_SUB)
    save(img, "fig_3_2_feature_vector.png")


def fig_3_3() -> None:
    horizontal_flow(
        "fig_3_3_training_sequence.png",
        "Рисунок 3.3 - Последовательность обучения пользовательского словаря жестов",
        [
            "Запись\nsample_*.npy",
            "Синхронизация\nс БД",
            "Загрузка\nдатасета",
            "Обучение\nKNN",
            "knn.pkl\nclasses.json",
            "Обновление\nmodel_class_id",
        ],
        "После записи сэмплы доступны и на диске, и через ORM-метаданные",
        [(GREEN_LIGHT, GREEN), (BLUE_LIGHT, BLUE), (ORANGE_LIGHT, ORANGE), (PURPLE_LIGHT, PURPLE), (RED_LIGHT, RED)],
    )


def fig_3_4() -> None:
    horizontal_flow(
        "fig_3_4_command_execution_flow.png",
        "Рисунок 3.4 - Поток выполнения команды после распознавания жеста",
        [
            "Метка\nжеста",
            "Confidence",
            "Политика\nthreshold",
            "Cooldown",
            "Command\nexecutor",
            "Действие\nmacOS",
            "Журнал",
        ],
        "Команда выполняется только после проверок безопасности",
        [(BLUE_LIGHT, BLUE), (GREEN_LIGHT, GREEN), (ORANGE_LIGHT, ORANGE), (ORANGE_LIGHT, ORANGE), (PURPLE_LIGHT, PURPLE), (RED_LIGHT, RED), ("#ffffff", LINE)],
    )


def fig_3_5() -> None:
    img, d = canvas("Рисунок 3.5 - Основные экраны пользовательского интерфейса GestureBind", "Рабочие зоны приложения и переходы")
    shell = (95, 170, 1705, 920)
    d.rounded_rectangle(shell, radius=32, fill="#ffffff", outline=LINE, width=4)
    d.rectangle((95, 170, 1705, 260), fill=BLUE_LIGHT)
    d.text((130, 195), "Глобальный статус: камера, модель, БД, автоисполнение", fill=INK, font=F_SUB)
    d.rounded_rectangle((130, 315, 430, 860), radius=24, fill="#f2f5f9", outline=LINE, width=3)
    for i, item in enumerate(["Главная", "Обучение", "Жесты", "Привязки", "Настройки"]):
        y = 350 + i * 90
        small_box(d, (160, y, 400, y + 58), item, fill="#ffffff", outline=BLUE if i == 0 else LINE, chars=14)
    box(d, (520, 330, 960, 575), "Live-превью\nметка + confidence", fill=GREEN_LIGHT, outline=GREEN, chars=20, f=F_SMALL)
    box(d, (1060, 330, 1580, 575), "Рабочая форма\nвыбранного экрана", fill=ORANGE_LIGHT, outline=ORANGE, chars=22, f=F_SMALL)
    box(d, (520, 650, 1580, 830), "Журнал событий и диагностические сообщения", fill="#ffffff", outline=LINE, chars=34, f=F_SMALL)
    save(img, "fig_3_5_ui_screens.png")


def fig_4_1() -> None:
    img, d = canvas("Рисунок 4.1 - Схема экранов приложения GestureBind", "Постоянная боковая навигация")
    center = (900, 520)
    box(d, (730, 430, 1070, 590), "Shell\nнавигация + статус", fill=BLUE_LIGHT, outline=BLUE, chars=18, f=F_SMALL)
    nodes = [
        ("Главная", 900, 210, GREEN_LIGHT, GREEN),
        ("Обучение", 1350, 360, ORANGE_LIGHT, ORANGE),
        ("Жесты", 1350, 700, PURPLE_LIGHT, PURPLE),
        ("Привязки", 450, 700, RED_LIGHT, RED),
        ("Настройки", 450, 360, "#ffffff", LINE),
    ]
    for text, cx, cy, fill, outline in nodes:
        box(d, (cx - 165, cy - 70, cx + 165, cy + 70), text, fill=fill, outline=outline, chars=12, f=F_SMALL)
        arrow(d, center, (cx, cy))
    save(img, "fig_4_1_screen_map.png")


def fig_4_2() -> None:
    img, d = canvas("Рисунок 4.2 - Общая структура окна приложения GestureBind", "Компонентная структура Flet-окна")
    d.rounded_rectangle((130, 165, 1670, 930), radius=28, fill="#ffffff", outline=LINE, width=4)
    d.rectangle((130, 165, 1670, 250), fill=BLUE_LIGHT)
    d.text((170, 194), "Верхняя панель: название, статус, быстрые индикаторы", fill=INK, font=F_SUB)
    d.rounded_rectangle((165, 290, 440, 880), radius=22, fill="#f2f5f9", outline=LINE, width=3)
    d.text((205, 318), "Навигация", fill=INK, font=F_SUB)
    d.rounded_rectangle((500, 290, 1625, 880), radius=22, fill="#ffffff", outline=BLUE, width=4)
    d.text((540, 318), "Область контента выбранного экрана", fill=INK, font=F_SUB)
    small_box(d, (540, 405, 910, 560), "Основная форма\nили видеопоток", fill=GREEN_LIGHT, outline=GREEN)
    small_box(d, (980, 405, 1560, 560), "Панели параметров\nи действий", fill=ORANGE_LIGHT, outline=ORANGE)
    small_box(d, (540, 650, 1560, 805), "Лог, подсказки, предупреждения, последние события", fill="#ffffff", outline=LINE, chars=34)
    save(img, "fig_4_2_window_structure.png")


def fig_5_1() -> None:
    img, d = canvas("Рисунок 5.1 - Общая схема тестирования GestureBind", "Автоматизированные проверки + ручные сценарии")
    box(d, (690, 180, 1110, 320), "Цель проверки\nработоспособность системы", fill=BLUE_LIGHT, outline=BLUE, chars=22, f=F_SMALL)
    branches = [
        ("Unit-тесты\nсервисы, ORM,\nполитики", 160, 500, GREEN_LIGHT, GREEN),
        ("Интеграция\nзапуск, модель,\nБД", 600, 500, ORANGE_LIGHT, ORANGE),
        ("Ручные сценарии\nкамера, GUI,\nкоманды", 1040, 500, PURPLE_LIGHT, PURPLE),
        ("Метрики\naccuracy,\nprecision, recall", 1440, 500, RED_LIGHT, RED),
    ]
    for text, cx, cy, fill, outline in branches:
        box(d, (cx - 170, cy - 95, cx + 170, cy + 95), text, fill=fill, outline=outline, chars=18, f=F_SMALL)
        arrow(d, (900, 320), (cx, cy - 95))
    box(d, (550, 820, 1250, 960), "Итог: подтверждение требований и выявление ограничений", fill="#ffffff", outline=LINE, chars=36, f=F_SMALL)
    for _text, cx, cy, *_ in branches:
        arrow(d, (cx, cy + 95), (900, 820))
    save(img, "fig_5_1_testing_scheme.png")


def fig_5_2() -> None:
    horizontal_flow(
        "fig_5_2_user_test_sequence.png",
        "Рисунок 5.2 - Последовательность пользовательского тестового сценария",
        [
            "Запуск",
            "Запись\nжеста",
            "Обучение",
            "Импорт\nв БД",
            "Привязка",
            "Распознавание",
            "Удаление\nсэмплов",
        ],
        "Сценарий проверяет полный цикл пользовательского жеста",
        [(BLUE_LIGHT, BLUE), (GREEN_LIGHT, GREEN), (ORANGE_LIGHT, ORANGE), (PURPLE_LIGHT, PURPLE), (RED_LIGHT, RED)],
    )


def fig_5_3() -> None:
    img, d = canvas(
        "Рисунок 5.3 - Сравнение вариантов обработки признаков на локальном наборе CTRLZ/New",
        "Метрика Accuracy, 5-fold cross-validation, KNN k=5",
    )
    labels = ["Базовый\npipeline", "Стандартизация", "Центрирование\nпримера", "L2-\nнормализация"]
    values = [0.926829268292683, 0.926829268292683, 1.0, 0.9024390243902439]
    x0, y0 = 260, 800
    plot_w, plot_h = 1250, 500
    d.line((x0, y0, x0 + plot_w, y0), fill=INK, width=4)
    d.line((x0, y0, x0, y0 - plot_h), fill=INK, width=4)
    for tick in [0.8, 0.85, 0.9, 0.95, 1.0]:
        y = y0 - int((tick - 0.8) / 0.2 * plot_h)
        d.line((x0 - 12, y, x0 + plot_w, y), fill="#d6dbe3", width=2)
        d.text((100, y - 16), f"{tick:.2f}", fill=MUTED, font=F_TINY)
    bar_w = 185
    gap = 105
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = x0 + 110 + i * (bar_w + gap)
        h = int((value - 0.8) / 0.2 * plot_h)
        d.rounded_rectangle((x, y0 - h, x + bar_w, y0), radius=18, fill=colors[i])
        d.text((x + 30, y0 - h - 44), f"{value:.3f}", fill=INK, font=F_SMALL)
        centered_text(d, (x - 35, y0 + 25, x + bar_w + 35, y0 + 130), label, fill=INK, f=F_TINY, chars=14)
    d.text((80, 255), "Accuracy", fill=INK, font=F_SMALL)
    d.text((1110, 980), "Вариант обработки признаков", fill=MUTED, font=F_TINY)
    save(img, "fig_5_3_metrics_accuracy.png")


def main() -> None:
    fig_1_1()
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_3_1()
    fig_3_2()
    fig_3_3()
    fig_3_4()
    fig_3_5()
    fig_4_1()
    fig_4_2()
    fig_5_1()
    fig_5_2()
    fig_5_3()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
