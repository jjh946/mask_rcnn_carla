import os
import argparse

import torch
import torchvision

from torchvision.transforms.v2 import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# import utils and train utils from references folder
from references.detection.engine import train_one_epoch, evaluate

# import tensorboard
import torch.utils.tensorboard as tb

import dataset_class2

# get the arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='NIA Train Script')
    parser.add_argument('--data_path', type=str, required=True, help='The absolute root path of the image')
    parser.add_argument('--val_path', type=str, required=True, help='The absolute output path') 
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='The number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='The learning rate for training')

    return parser.parse_args()

# define maskrcnn model
def get_model_instance_segmentation(num_classes):
    #load pretrained model 
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    #replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# preprocess the image and annotations
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


#define the collate function
def collate_fn(batch):
    return tuple(zip(*batch))



def train_script(data_path, val_path, batch_size, num_epochs, learning_rate):
    img_dir = os.path.join(data_path, 'image')   
    label_dir = os.path.join(data_path, 'label_json') 

    val_img_dir = os.path.join(val_path, 'image')
    val_label_dir = os.path.join(val_path, 'label_json')

    # logging path
    log_path = os.path.commonprefix([img_dir, val_img_dir])
    log_path = os.path.join(log_path, 'result','logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Log path created at {log_path}")

    # logging
    log_writer = tb.SummaryWriter(log_path)

    # metrics for logging
    metrics = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large']

    # define the dataset
    train_dataset = dataset_class2.InstanceSegmentationDataset(img_dir, label_dir, get_transform(train=True))
    val_dataset = dataset_class2.InstanceSegmentationDataset(val_img_dir, val_label_dir, get_transform(train=False))

    # define the data loader
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    val_data_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # define the model
    num_classes = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        )
    
    # construct a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    #start training
    best_score = -9999
    best_model = None

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_log = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10, logging_path=log_path)
        for k,v in train_log.meters.items():
            log_writer.add_scalar("train_loss/" + k, v.value, epoch)
        
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        test_log = evaluate(model, val_data_loader, device=device)
        
        # log the metrics
        bbox_metrics = test_log.coco_eval['bbox'].stats
        segm_metrics = test_log.coco_eval['segm'].stats

        for idx,metric in enumerate(metrics):
            log_writer.add_scalar("val_metrics_bbox/" + metric, bbox_metrics[idx],epoch)
            log_writer.add_scalar("val_metrics_segm/" + metric, segm_metrics[idx], epoch)
        
        if bbox_metrics[0] > best_score:
            best_score = bbox_metrics[0]
            best_model = model
            model_path = '/'.join(log_path.split('/')[:-1])
            # save as torch script
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(model_path, 'best_mask_rcnn_model.pt'))
            print(f"Best model saved at {os.path.join(model_path, f'best_mask_rcnn_model.pth')}")

def main():
    args = parse_arguments()
    # print(args)
    train_script(
        args.data_path, 
        args.val_path, 
        args.batch_size, 
        args.num_epochs, 
        args.learning_rate
        )

if __name__ == "__main__":
    main()