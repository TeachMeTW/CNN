# app/Dockerfile


FROM python:3.11

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Data_Analysis_App_Wireframe.py", "--server.port=8501", "--server.address=0.0.0.0"]