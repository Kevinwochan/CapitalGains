FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libsqlite3-dev \
    && pip install pipx \
    && rm -rf /var/lib/apt/lists/*

# Ensure pipx binaries are on the PATH
ENV PATH="$PATH:/root/.local/bin"

RUN pipx install uv

COPY ./app/requirements.txt requirements.txt

RUN uv pip install --system -r requirements.txt

COPY ./app /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
