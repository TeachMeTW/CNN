# CNN 

# Ronan Was Here
A Streamlit App

Local Setup Instructions

1. Download Docker

   - Visit the Docker website to download and install Docker on your system: https://www.docker.com/products/docker-desktop

2. Build the Docker Image

   - In your project directory, run the following command to build the Docker image:

     `docker build -t streamlit .`

3. Run the Docker Container

   - To run the container and expose it on port 8501, use the following command:

     `docker run -p 8501:8501 streamlit`
