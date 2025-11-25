#!/bin/bash
# test_in_docker.sh - build and test in exact docker env
# Location: ~/CS224W_Project/test_in_docker.sh

set -e

PROJECT_ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"
echo "Building Docker image from Dockerfile..."
docker build -t cs224w-test:latest .

echo ""
echo "Running tests inside Docker container..."
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    -e GCS_BUCKET=${GCS_BUCKET:-default-bucket-name} \
    cs224w-test:latest \
    bash tests/run_local_test.sh

echo ""
echo "Tests completed successfully in Docker env!"
EOF

chmod +x test_in_docker.sh