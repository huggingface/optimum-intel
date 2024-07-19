#!/bin/bash

# # Define image name
IMAGE_NAME="myimage:latest"

# Step 1: Build the Docker image
docker build -f Dockerfile.intel -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -ne 0 ]; then
  echo "Docker image build failed."
  exit 1
fi

# Step 2: Run a container from the built image in detached mode with an override command
CONTAINER_ID=$(docker run -d $IMAGE_NAME tail -f /dev/null)

# Step 3: Check if the container is running
if docker inspect -f '{{.State.Running}}' $CONTAINER_ID 2>/dev/null | grep -q 'true'; then
  echo "Container is running."
else
  echo "Container failed to start."
  docker logs $CONTAINER_ID 2>/dev/null || echo "No container ID found."
  exit 1
fi

# Step 4: Clean up - stop and remove the container
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo "Sanity test completed successfully."