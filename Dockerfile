# Trail MCP server image — serves the six Trail tools over streamable-HTTP (endpoint /mcp).
# Built from the local trail-lang source; the [mcp] extra pulls the MCP SDK.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Package metadata + source (hatchling builds the `trail` package, incl. trail/stdlib/*.trail).
COPY pyproject.toml README.md ./
COPY trail ./trail
RUN pip install ".[mcp]"

# Non-root runtime user.
RUN useradd --create-home --uid 1000 trail
USER trail

EXPOSE 3000
# Streamable-HTTP MCP endpoint at http://<host>:3000/mcp
ENTRYPOINT ["trail", "mcp"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "3000"]
