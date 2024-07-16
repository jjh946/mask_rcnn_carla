import base64
import numpy as np

def decode_polygon(base64_string, width=1280, height=720):
    # Base64 디코딩
    binary_data = base64.b64decode(base64_string)
    
    # 바이너리 데이터를 uint32 배열로 변환
    rle_values = []
    for i in range(0, len(binary_data), 4):
        value = int.from_bytes(binary_data[i:i+4], byteorder='little')
        rle_values.append(value)
    
    # RLE 데이터를 이진 이미지로 변환
    binary_mask = []
    for i, length in enumerate(rle_values):
        value = i % 2  # RLE에서 짝수 인덱스는 0, 홀수 인덱스는 1로 번갈아가며 나타내므로 이를 이진 값으로 변환
        binary_mask.extend([value] * length)
    
    # 이미지 배열로 변환하여 반환
    binary_mask = np.array(binary_mask, dtype=np.uint8)
    binary_mask = binary_mask.reshape((height, width))

    # 이진 마스크에서 좌표 추출
    pointsyx = np.transpose(np.nonzero(binary_mask))
    pointsyx_list = [tuple(point) for point in pointsyx]
    
    return pointsyx_list


def polygon_to_bounding_box(polygon_points):
    x_coords = [point[0] for point in polygon_points]
    y_coords = [point[1] for point in polygon_points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    return [x_min, y_min, x_max, y_max]
