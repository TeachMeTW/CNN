# app/Dockerfile


FROM python:3.11.5

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN mkdir -p /ml

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -r ./ml/requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
