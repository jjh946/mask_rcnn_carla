import base64
import numpy as np
from PIL import Image, ImageDraw
import os
import json
import torch
from torch.utils.data import Dataset

def decode_polygon(base64_string, width=1280, height=720):
    binary_data = base64.b64decode(base64_string)
    rle_values = []
    for i in range(0, len(binary_data), 4):
        value = int.from_bytes(binary_data[i:i+4], byteorder='little')
        rle_values.append(value)
    
    binary_mask = []
    for i, length in enumerate(rle_values):
        value = i % 2
        binary_mask.extend([value] * length)
    
    if len(binary_mask) != width * height:
        raise ValueError(f"Cannot reshape array of size {len(binary_mask)} into shape ({height},{width})")
    
    binary_mask = np.array(binary_mask, dtype=np.uint8)
    binary_mask = binary_mask.reshape((height, width))
    pointsyx = np.transpose(np.nonzero(binary_mask))
    pointsyx_list = [tuple(point) for point in pointsyx]
    pointsyx_list = [(point[1], point[0]) for point in pointsyx]  # 여기서 (y, x) -> (x, y)로 변경

    return pointsyx_list

def polygon_to_bounding_box(polygon_points):
    if not polygon_points:
        return [0, 0, 0, 0]
    x_coords = [point[0] for point in polygon_points]
    y_coords = [point[1] for point in polygon_points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    return [x_min, y_min, x_max, y_max]

class InstanceSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.filenames = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transforms = transform
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx] + '.png')
        annotation_path = os.path.join(self.annotation_dir, self.filenames[idx] + '.json')

        image = Image.open(image_path).convert("RGB")

        with open(annotation_path) as f:
            annotations = json.load(f)

        label_dict = {
            "Vehicles": 1,
            "Pedestrian": 2,
            "TrafficLight": 3,
            "TrafficSigns": 4,
        }

        mask_annotations = annotations['annotations']

        masks = []
        labels = []
        boxes = []
        is_crowd = []
        
        for mask_annotation in mask_annotations:  
            mask_img = Image.new('L', image.size, 0)
            polygon_point_list = decode_polygon(mask_annotation['points'], width=image.width, height=image.height)
            
            ImageDraw.Draw(mask_img).polygon(polygon_point_list, outline=1, fill=1)
            mask = np.array(mask_img)
            masks.append(mask)

            box = polygon_to_bounding_box(polygon_point_list)
            boxes.append(box)
            
            mask_label = mask_annotation['class']
            labels.append(label_dict.get(mask_label, 0))  # Ensure label is within expected range

        if masks:
            masks = np.stack(masks, axis=-1)
        else:
            masks = np.zeros((image.height, image.width, 1))
        
        image = np.array(image).transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.uint8)
        
        masks = masks.transpose(2, 0, 1)
        masks = torch.tensor(masks, dtype=torch.uint8)

        image_id = idx
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.int32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.int32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target
