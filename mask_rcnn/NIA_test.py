import os
import argparse
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T


# import utils and train utils from references folder
from references.detection.engine import evaluate

# import dataset class
import dataset_class2


# get arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='NIA Train Script')
    parser.add_argument('--test_data_path', type=str, required=True, help='The absolute root path of the test data path')
    parser.add_argument('--model_path', type=str, required=True, help='The absolute path for model') 

    return parser.parse_args()

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def inference(model, test_data_path):
    # load the test data
    img_dir = os.path.join(test_data_path, 'image')
    annotation_dir = os.path.join(test_data_path, 'label_json')
    test_dataset = dataset_class2.InstanceSegmentationDataset(img_dir, annotation_dir, get_transform(train=False))
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=collate_fn)

    # evaluate the model
    evaluate(model, test_data_loader, device=torch.device('cuda'), saved_model=True)

def predict_imgs(model, test_data_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = read_image(test_data_path)
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[1][0]

    label_dict = {"Vehicles":1,"Pedestrian":2,"TrafficLight":3,"TrafficSigns":4}
    reversed_label_dict = {v: k for k, v in label_dict.items()}

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # labels indices whose score is greater than 0.5
    valueable_labels = [i for i, score in enumerate(pred["scores"]) if score > 0.5]

    # only keep the labels dict whose score is greater than 0.5
    pred_labels = [f"{reversed_label_dict[int(label.cpu())]} : {score:.3f}" for label, score in zip(pred["labels"][valueable_labels], pred['scores'][valueable_labels])]
    
    # preserve the boxes whose score is greater than 0.5
    pred_boxes = pred["boxes"].long()[valueable_labels]
    # draw the bounding boxes
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # preserve the masks whose score is greater than 0.5
    masks = (pred["masks"] > 0.5).squeeze(1)[valueable_labels]
    # draw the masks
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")
    
    # save the result
    img_name = os.path.basename(test_data_path)
    save_path = './Output/result/image'
    print(f"test image : [{img_name}] is being saved")
    torchvision.io.write_png(output_image, os.path.join(save_path, img_name))


def main():
    import sys
    args = parse_arguments()
    print("start testing")

    # load the model
    model = torch.jit.load(args.model_path)
    model.to('cuda')

    # load the test data
    test_data_path = args.test_data_path

    # test the model
    inference(model, test_data_path)

    # predict images and save the result
    img_paths = [os.path.join(test_data_path, 'image', img) for img in os.listdir(os.path.join(test_data_path, 'image'))]
    for img_path in img_paths:
        predict_imgs(model, img_path)

if __name__ == "__main__":
    main()
    
