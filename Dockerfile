# app/Dockerfile

FROM python:3.11

# Set working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the Streamlit example into the /app directory (if needed)
RUN git clone https://github.com/streamlit/streamlit-example.git .

# Copy the main folder (where streamlit_app.py is located) into the container
COPY ./main /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Add healthcheck to verify if the Streamlit server is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
