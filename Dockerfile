FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsqlite3-dev \
    && pip install pipx \
    && rm -rf /var/lib/apt/lists/*

# Ensure pipx binaries are on the PATH
ENV PATH="$PATH:/root/.local/bin"

RUN pipx install uv

# Create virtual environment
RUN uv venv /venv/

# Activate virtual environment by setting PATH
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY ./app/requirements.txt requirements.txt

# Install packages into virtual environment
RUN uv pip install -r requirements.txt

COPY ./app /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
