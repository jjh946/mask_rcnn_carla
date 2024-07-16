
from torch.utils.data import DataLoader
from dataset_class2 import InstanceSegmentationDataset

# 테스트할 디렉토리 경로
image_dir = './Output/train/image/'
annotation_dir = './Output/train/label_json/'

# 데이터셋 인스턴스 생성
dataset = InstanceSegmentationDataset(image_dir, annotation_dir)

# DataLoader 생성
data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 데이터 로드 및 검증
for images, targets in data_loader:
    print(f'Images shape: {images.shape}')
    print(f'Targets: {targets}')
    break  # 첫 배치만 확인
