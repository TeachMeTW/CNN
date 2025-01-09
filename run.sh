#!/bin/bash

# Detect host CPU architecture
HOST_ARCH=$(uname -m)

# Map host architecture to image tag convention
case "$HOST_ARCH" in
  x86_64)
    export ARCH=amd64
    ;;
  aarch64|arm64)
    export ARCH=arm64
    ;;
  *)
    echo "Unsupported architecture: $HOST_ARCH"
    exit 1
    ;;
esac

# Optionally, you can print which architecture was detected
echo "Detected architecture: $ARCH"

# Run Docker Compose with the environment set
docker-compose -f docker-compose-dev.yml up 
