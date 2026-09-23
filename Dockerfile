# The public demo: the tagger, without the vision engine.
#
# Only match_tag/requirements.txt is installed. requirements-cv.txt pulls in
# torch, ultralytics and opencv — well over a gigabyte, and useless here since
# the demo refuses to run an analysis anyway. The engine is imported lazily by
# match_tag/analysis.py, so the app starts and serves without it and reports
# the engine as unavailable.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PUBLIC_DEMO=1

WORKDIR /app

COPY match_tag/requirements.txt ./match_tag/requirements.txt
RUN pip install --no-cache-dir -r match_tag/requirements.txt

COPY . .

WORKDIR /app/match_tag

EXPOSE 5055

# Exactly one worker. The JobRunner that tracks analysis jobs lives in the
# process, so a second worker would hold its own copy and answer status polls
# about jobs it has never heard of. Threads give the concurrency instead.
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5055", "app:app"]
