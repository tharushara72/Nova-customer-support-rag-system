FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (Docker layer caching — if these don't change, deps won't reinstall)
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile (no dev tools, exact versions)
RUN uv sync --frozen --no-dev

# Copy all source code
COPY . .

# Create runtime directories
RUN mkdir -p artifacts logs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]