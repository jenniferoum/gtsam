#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Docker Hub username
DOCKER_USERNAME="borglab"
DOCKER_REPOSITORY="gtsam-ci"

# Define arrays for Ubuntu versions and compilers
declare -a build_configs=( 
  "ubuntu 22.04 gcc 9"
  "ubuntu 22.04 clang 11"
  "ubuntu 22.04 clang 14"
  "ubuntu 24.04 gcc 14"
  "ubuntu 24.04 clang 16"
)

for config in "${build_configs[@]}"; do
  # Split the config string into variables
  read -r os os_version compiler compiler_version <<< "$config"
  
  tag="${os}-${os_version}-${compiler}-${compiler_version}"
  dockerfile_path="Dockerfile.${os}-${compiler}"

  # Check if Dockerfile exists
  if [ -f "$dockerfile_path" ]; then
    echo "Building Docker image for $tag"
    docker build -t "$DOCKER_USERNAME/$DOCKER_REPOSITORY:$tag" -f "$dockerfile_path" \
      --build-arg UBUNTU_VERSION="$os_version" \
      --build-arg COMPILER_VERSION="$compiler_version" .
    
    echo "Pushing Docker image $DOCKER_USERNAME/$DOCKER_REPOSITORY:$tag"
    docker push "$DOCKER_USERNAME/$DOCKER_REPOSITORY:$tag"
    
    echo "Successfully built and pushed $DOCKER_USERNAME/$DOCKER_REPOSITORY:$tag"
  else
    echo "Warning: Dockerfile not found at $dockerfile_path, skipping..."
  fi
done
 
echo "All Docker images processed."