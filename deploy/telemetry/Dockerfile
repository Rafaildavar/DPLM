FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TELEMETRY_HOST=0.0.0.0 \
    TELEMETRY_PORT=8787 \
    TELEMETRY_DATABASE=/data/telemetry.sqlite

WORKDIR /srv/gesturebind

RUN useradd --create-home --uid 10001 telemetry \
    && mkdir -p /data app/services scripts \
    && chown -R telemetry:telemetry /data /srv/gesturebind

COPY --chown=telemetry:telemetry app/__init__.py app/__init__.py
COPY --chown=telemetry:telemetry app/services/__init__.py app/services/__init__.py
COPY --chown=telemetry:telemetry app/services/telemetry_collector.py app/services/telemetry_collector.py
COPY --chown=telemetry:telemetry scripts/__init__.py scripts/__init__.py
COPY --chown=telemetry:telemetry scripts/telemetry_collector.py scripts/telemetry_collector.py

USER telemetry
EXPOSE 8787
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2)"]

CMD ["python", "-m", "scripts.telemetry_collector"]
