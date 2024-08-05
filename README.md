# Mask R CNN for Carla Data

이 문서의 모든 명령어는 이 프로젝트의 위치 `mask_rcnn_carla` 에서 실행된다.

# 자동화 실행

사용자가 도커 이미지를 빌드하거나, 컨테이너 내부로 들어가서 명령어를 실행시키기 어렵다면, `just_train`과 `just_test` 프로그램을 이용해서 간단하게 데이터를 학습시키고, 성능을 테스트 해볼 수 있다. 학습된 모델은 도커 컨테이너 밖의 `final_result` 디렉토리에 저장된다.

## Just Train

사용자가 단지 데이터를 학습을 시키고 싶은 것이라면, 학습시키고 싶은 데이터가 있는 위치를 입력 해서 다음 코드를 실행시켜주면 된다. 다음 코드는 도커 컨테이너를 빌드하고 실행하여, mask r cnn 데이터를 학습시킨다.

```bash
./just_train.sh [data의 상위 디렉토리]
```

예를 들어 데이터의 구조가 다음과 같다면,

```bash
saved
├── data
    ├── labelingData
    └── rawData
```

다음과 같이 명령어를 실행시켜주면 된다.

```bash
./just_train.sh /saved
```

디렉토리 구조가 다음과 같아도 모든 데이터를 순회하며 학습을 하도록 설계되었다.

```bash
saved
├── data1
│   ├── labelingData
│   └── rawData
└── data2
    ├── labelingData
    └── rawData

```

## Just Test

`just_train`을 실행시켰다면, `vehicle_detection`이라는 이름의 이미지를 가진 컨테이너가 하나 생성되었을 것이다. `just_test`는 이 컨테이너를 실행시켜, 학습된 모델의 성능을 테스트하고 테스트된 이미지를 `final_result`에 저장해준다

```bash
./just_test.sh
```

# 세부 실행

아래 내용은 직접 도커 이미지를 빌드하고 컨테이너를 관리하는 과정을 보여준다. 위의 `just_train`과 `just_test` 에서는 이미지의 이름을 모두 `vehicle_detection`으로 하였다. 원한다면 이미지의 이름을 바꾸어줘도 무방하다.

### 도커 빌드

```bash
docker build -t [이미지 이름] .
```

```bash
docker build -t vehicle_detection .
```

### 도커 실행 및 학습 시작

도커 파일을 실행시키면 자동으로 학습을 시작하도록 되어있다. 최종 결과를 받을 디렉토리와 학습 시킬 데이터의 위치를 입력해주어야 한다.

```bash
docker run -it --gpus all -v [final_output 위치]:/workspace/Output/result -v [data의 상위 디렉토리]:/workspace/data [이미지 이름]
```

```bash
docker run -it --gpus all -v ~/Desktop/workspace/maskrcnn/final_output:/workspace/Output/result -v ~/Desktop/workspace/saved:/workspace/data vehicle_detection
```

지금까지의 과정에서 생성된 도커 컨테이너(예시 컨테이너 ID: `992e4a6068e9`)를 다시 실행시키기 위해서는, 해당 컨테이너가 정지된 상태이어야 하며, 삭제되지 않아야 한다. 도커에서는 컨테이너가 중지된 상태라면 언제든지 다시 시작할 수 있다.

### 이전에 생성된 컨테이너 실행하기

1. **중지된 컨테이너 확인**
    
    먼저, `docker ps -a` 명령어를 사용하여 중지된 컨테이너를 포함한 모든 컨테이너 목록을 확인한다. 여기에서 `992e4a6068e9` ID를 가진 컨테이너가 있어야 한다.
    
    ```bash
    docker ps -a
    ```
    
    이 명령어는 현재 실행 중인 컨테이너뿐만 아니라, 중지된 컨테이너도 모두 나열한다. `992e4a6068e9`가 목록에 있는지 확인한다.
    
2. **컨테이너 재시작**
    
    `docker start` 명령어를 사용하여 중지된 컨테이너를 다시 시작할 수 있다.
    
    ```bash
    docker start 992e4a6068e9
    ```
    
3. **컨테이너에 접속**
    
    컨테이너가 시작된 후, `docker exec` 명령어를 사용하여 해당 컨테이너에 접속하고 명령을 실행할 수 있다.
    
    ```bash
    docker exec -it 992e4a6068e9 /bin/bash
    ```
    
    이 명령어는 컨테이너의 셸로 접속하여, 내부에서 명령을 실행할 수 있도록 한다.
    

### 추가 옵션

- **자동으로 셸로 접속하고 실행**: `docker start` 명령어를 `ai` 옵션과 함께 사용하여 컨테이너를 시작하면서 동시에 셸로 접속할 수 있다.
    
    ```bash
    docker start -ai 992e4a6068e9
    ```
    
    이 명령어는 컨테이너를 시작하고 그 콘솔로 바로 연결해준다.
    
- **컨테이너 내부에서 명령 실행**: 컨테이너가 실행된 상태라면 내부에서 직접 원하는 명령을 실행할 수 있다. 예를 들어:
    
    ```bash
    python3 /workspace/NIA_test.py --test_data_path /workspace/Output/test/ --model_path /workspace/Output/result/best_mask_rcnn_model.pt
    ```
