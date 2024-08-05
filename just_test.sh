#!/bin/bash

# 설정할 이미지 이름
IMAGE_NAME="vehicle_detection"

# 이미지 이름을 기반으로 가장 최근에 생성된 컨테이너 ID 찾기
CONTAINER_ID=$(docker ps -a --filter "ancestor=$IMAGE_NAME" --format "{{.ID}}" | head -n 1)

# 컨테이너가 존재하는지 확인
if [ -z "$CONTAINER_ID" ]; then
  echo "No container found for image $IMAGE_NAME"
  exit 1
fi


# 컨테이너 시작
docker start $CONTAINER_ID

# 컨테이너 내부에서 스크립트 실행
docker exec -it $CONTAINER_ID /bin/bash -c "
  python3 /workspace/NIA_test.py --test_data_path /workspace/Output/test/ --model_path /workspace/Output/result/best_mask_rcnn_model.pt
"