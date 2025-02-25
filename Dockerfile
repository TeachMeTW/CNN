# Stage 1: Builder Stage
FROM python:3.11-slim-bullseye AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Override libc-bin post-installation script to avoid QEMU segfaults
RUN echo '#!/bin/sh\nexit 0' > /var/lib/dpkg/info/libc-bin.postinst && \
    chmod +x /var/lib/dpkg/info/libc-bin.postinst

# Install only the build dependencies needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/* && \
    dpkg --configure -a

# Copy just the requirements to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies without caching pip packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Runtime Stage
FROM python:3.11-slim-bullseye
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
EXPOSE 8501

# Override libc-bin post-installation script in runtime stage as well
RUN echo '#!/bin/sh\nexit 0' > /var/lib/dpkg/info/libc-bin.postinst && \
    chmod +x /var/lib/dpkg/info/libc-bin.postinst

# Install only runtime libraries using the proper package names
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103 \
    libfreetype6 \
    libpng16-16 && \
    rm -rf /var/lib/apt/lists/* && \
    dpkg --configure -a

# Copy the installed Python packages and executables from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application code
COPY . .

# Start the Streamlit app
ENTRYPOINT ["streamlit", "run", "Overview.py", "--server.port=8501", "--server.address=0.0.0.0"]
