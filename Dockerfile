# tokenroute — Track 1 routing agent. Stdlib only: no pip install step.
FROM python:3.12-slim

WORKDIR /app
COPY tokenroute/ tokenroute/
COPY eval/ eval/

# Config comes from the environment at run time:
#   LOCAL_MODEL, REMOTE_MODEL, OLLAMA_URL, FIREWORKS_API_KEY, ESCALATE_THRESHOLD
ENTRYPOINT ["python", "-m", "tokenroute.agent"]
CMD ["--help"]
