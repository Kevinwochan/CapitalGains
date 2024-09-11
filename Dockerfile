FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pipenv

COPY ./app/Pipfile Pipfile
COPY ./app/Pipfile.lock Pipfile.lock

RUN pipenv install

COPY ./app /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["pipenv", "run", "streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]