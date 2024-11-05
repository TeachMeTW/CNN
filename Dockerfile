# app/Dockerfile

# Use Python 3.10-slim as the base image for better compatibility and smaller image size
# Update to 3.11.5 for ML training 
FROM python:3.11.5

# Set environment variables to prevent Python from writing .pyc files and to buffer outputs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the Streamlit port
EXPOSE 8501

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    git \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

RUN mkdir -p /ml

# Copy the requirements file into the container first for better caching
COPY requirements.txt .
COPY ./ml/requirements.txt ./ml/requirements.txt

# Upgrade pip and install Python dependencies 
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
# Install the ml requirements
RUN pip3 install -r ./ml/requirements.txt

# Copy the rest of the application code
COPY . .

# Define the default command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
