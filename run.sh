#!/bin/bash
set -e

# --- Helper Functions ---

# Check if a command exists in PATH
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Install Docker Engine (Linux only)
install_docker() {
  echo "Installing Docker..."

  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID=$ID
  else
    echo "Cannot detect OS distribution (missing /etc/os-release). Exiting."
    exit 1
  fi

  if [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    curl -fsSL https://download.docker.com/linux/${OS_ID}/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/${OS_ID} $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

  elif [[ "$OS_ID" == "centos" || "$OS_ID" == "rhel" || "$OS_ID" == "fedora" ]]; then
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum install -y docker-ce docker-ce-cli containerd.io

  else
    echo "Unsupported OS: $OS_ID"
    exit 1
  fi

  sudo systemctl start docker
  sudo systemctl enable docker
  echo "Docker installation complete."
}

# Install Docker Compose (Linux only; macOS/Windows users should use Docker Desktop)
install_docker_compose() {
  echo "Installing Docker Compose..."
  DOCKER_COMPOSE_VERSION="2.20.2"
  sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  echo "Docker Compose installation complete."
}

# --- Main Script ---

# Detect OS type
OS_TYPE=$(uname)
if [[ "$OS_TYPE" == "Darwin" ]]; then
  echo "Detected OS: macOS"
elif [[ "$OS_TYPE" == "Linux" ]]; then
  echo "Detected OS: Linux"
elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == MSYS* || "$OS_TYPE" == CYGWIN* ]]; then
  echo "Detected OS: Windows"
else
  echo "Unsupported OS: $OS_TYPE"
  exit 1
fi

# 1. Ensure Docker is installed.
if [[ "$OS_TYPE" == "Darwin" ]]; then
  if ! command_exists docker; then
    echo "Docker is not installed. Please install Docker Desktop for macOS from:"
    echo "https://www.docker.com/products/docker-desktop/"
    exit 1
  fi

elif [[ "$OS_TYPE" == "Linux" ]]; then
  if ! command_exists docker; then
    install_docker
  else
    echo "Docker is already installed."
  fi

elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == MSYS* || "$OS_TYPE" == CYGWIN* ]]; then
  if ! command_exists docker; then
    echo "Docker is not installed. Please install Docker Desktop for Windows from:"
    echo "https://www.docker.com/products/docker-desktop/"
    exit 1
  fi
fi

# 2. Ensure the Docker daemon is running.
if [[ "$OS_TYPE" == "Darwin" ]]; then
  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running. Starting Docker Desktop..."
    open -a Docker
    echo "Waiting for Docker to start..."
    while ! docker info >/dev/null 2>&1; do
      sleep 1
    done
    echo "Docker is up and running."
  fi

elif [[ "$OS_TYPE" == "Linux" ]]; then
  if ! systemctl is-active --quiet docker; then
    echo "Docker daemon is not running. Starting Docker..."
    sudo systemctl start docker
  fi

elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == MSYS* || "$OS_TYPE" == CYGWIN* ]]; then
  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running. Please start Docker Desktop for Windows."
    exit 1
  fi
fi

# 3. Ensure Docker Compose is installed or available.
# On macOS and Windows, Docker Desktop includes Docker Compose (v2) as a plugin.
if command_exists docker-compose; then
  COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  if [[ "$OS_TYPE" == "Linux" ]]; then
    install_docker_compose
    COMPOSE_CMD="docker-compose"
  else
    echo "Docker Compose is not installed. Please update your Docker Desktop installation."
    exit 1
  fi
fi

# 4. Detect host CPU architecture and map to image tag convention.
HOST_ARCH=$(uname -m)
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

echo "Detected architecture: $ARCH"

# 5. Run Docker Compose with the environment set.
$COMPOSE_CMD -f docker-compose-dev.yml up
