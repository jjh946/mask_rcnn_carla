#!/bin/bash

# 사용법 출력 함수
usage() {
  echo "Usage: $0 <saved_directory_path>"
  exit 1
}

# 인자로 받은 경로 변수
SAVED_PATH="$1"

# 인자가 없는 경우 사용법 출력
if [ -z "$SAVED_PATH" ]; then
  echo "Error: No path provided."
  usage
fi

# 경로가 존재하는지 확인
if [ ! -d "$SAVED_PATH" ]; then
  echo "Error: The provided path does not exist or is not a directory."
  exit 1
fi

# 도커 이미지 빌드
echo "Building Docker image..."
docker build -t vehicle_detection .

# 도커 컨테이너 실행
echo "Running Docker container..."
docker run -it --rm --gpus all \
  -v "$PWD/final_output:/workspace/Output/result" \
  -v "$SAVED_PATH:/workspace/data" \
  vehicle_detection
