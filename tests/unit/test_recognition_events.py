from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Command, Gesture
from app.services.recognition_events import (
    list_recent_recognition_events,
    record_recognition_event,
)


def test_record_recognition_event_links_gesture_and_lists_command():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        gesture = Gesture(label="palm", model_class_id=0, is_active=True)
        session.add(gesture)
        session.flush()
        session.add(
            Command(
                name="Прокрутка вниз",
                platform="all",
                gesture_id=gesture.id,
                is_active=True,
            )
        )
        session.commit()

        row = record_recognition_event(
            session,
            label="palm",
            confidence=0.91,
            executed=True,
        )

        assert row.id is not None
        assert row.gesture_id == gesture.id
        assert row.executed is True

        events = list_recent_recognition_events(session, limit=5)
        assert len(events) == 1
        assert events[0]["label"] == "palm"
        assert events[0]["commandName"] == "Прокрутка вниз"
        assert events[0]["executed"] is True
        assert events[0]["confidence"] == 0.91
    finally:
        session.close()


def test_record_recognition_event_keeps_unknown_label():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        row = record_recognition_event(
            session,
            label="unknown",
            confidence=0.4,
            executed=False,
        )

        assert row.gesture_id is None
        events = list_recent_recognition_events(session)
        assert events[0]["label"] == "unknown"
        assert events[0]["commandName"] == ""
        assert events[0]["executed"] is False
    finally:
        session.close()
