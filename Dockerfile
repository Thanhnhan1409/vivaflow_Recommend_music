FROM python:3.10-slim

# --- Cài đặt hệ thống ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Thiết lập biến môi trường ---
ENV POETRY_VERSION=1.8.2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# --- Cài Poetry ---
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# --- Copy pyproject để cài dependencies ---
COPY pyproject.toml poetry.lock* ./

# --- Cài dependencies từ poetry (bỏ qua lỗi nếu thiếu lock) ---
RUN poetry install --no-root || true

# --- Cài uvicorn thủ công ---
RUN pip install --no-cache-dir "uvicorn[standard]"

# --- Copy mã nguồn ---
COPY . .

# --- Mở cổng FastAPI ---
EXPOSE 8000

# --- Chạy bằng Uvicorn ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
