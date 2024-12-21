# app/Dockerfile

# Stage 1: Build Stage
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements files
COPY requirements.txt .
COPY ml/requirements.txt ./ml/

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r ./ml/requirements.txt

# Stage 2: Final Stage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-serial-dev \
    libfreetype6-dev \
    libpng-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENTRYPOINT ["streamlit", "run", "Data_Analysis_App_Wireframe.py", "--server.port=8501", "--server.address=0.0.0.0"]
