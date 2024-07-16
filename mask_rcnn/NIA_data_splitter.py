import os
import shutil
import random
import argparse
import re


# take root path of image and label
# take output path
# take ratio of train, val, test

# make output dir if not exist
# list all the image and label
# make the pair of image and label
# shuffle the pair
# split the pair into train, val, test
# copy the pair into the output dir, keeping directory structure of root path of image and label

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--data_path', type=str, required=True, help='The absolute root path of the image')
    parser.add_argument('--label_path', type=str, required=True, help='The absolute path to the annotation')
    parser.add_argument('--output_path', type=str, required=True, help='The absolute output path')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='The ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='The ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='The ratio of test set')

    return parser.parse_args()

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_train_val_test(output_path):
    make_dir(os.path.join(output_path, 'train'))
    make_dir(os.path.join(output_path, 'val'))
    make_dir(os.path.join(output_path, 'test'))
    make_dir(os.path.join(output_path, 'result'))

def make_image_label(output_path):
    output_paths = os.listdir(output_path)
    print(output_paths)
    for path in output_paths:
        make_dir(os.path.join(output_path,path, 'image'))
        if path != 'result':
            make_dir(os.path.join(output_path, path, 'label_json'))

def list_all_files(data_path, annotation_path):
    # Incorporate _find_pairs logic directly here)
    pairs = []
    for case in os.listdir(data_path):
        case_path = os.path.join(data_path, case)
        # check if the case path is a directory
        
        if not os.path.isdir(case_path):
            continue

        for design in os.listdir(case_path):
            design_path = os.path.join(case_path, design, 'Image_RGB')
            label_design_path = os.path.join(annotation_path, case,design, 'outputJson','polygon')

            if not os.path.isdir(design_path) or not os.path.isdir(label_design_path):
                continue

            images = os.listdir(design_path)
            # if there is no image in the design path, skip
            if len(images) == 0:
                continue
            for image in images:
                image_path = os.path.join(design_path, image)
                
                label_path = os.path.join(label_design_path, image.replace('.png', '.json'))
                
                if os.path.isfile(image_path) and os.path.isfile(label_path):
                    pairs.append((image_path, label_path))
    return pairs
    
def split_data(data_path, label_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    pairs = list_all_files(data_path, label_path)
    random.shuffle(pairs)
    total = len(pairs)

    # split the data length
    train = int(total * train_ratio)
    val = int(total * val_ratio)
    
    # split the pairs
    train_pairs = pairs[:train]
    val_pairs = pairs[train:train+val]
    test_pairs = pairs[train+val:]

    return train_pairs, val_pairs, test_pairs
    


def copy_and_move(pairs_of_data, output_path):
    data_path , label_path = pairs_of_data
    # print(pairs_of_data)
    path_pattern = re.compile(r'N\d{2}S\d{2}M\d{2}')
    data_path_match = path_pattern.search(data_path).group()
    label_path_match = path_pattern.search(label_path).group()
    prefix = data_path if data_path_match == label_path_match else None
    # print(prefix)

    data_file_name = os.path.basename(data_path)
    label_file_name = os.path.basename(label_path)

    # check the data and label path are valid
    assert prefix is not None, "The prefix of the data and label path is not the same"
    assert data_path.endswith('.png') and label_path.endswith('.json'), "The data and label path is not a valid file data: {} label: {}".format(data_path, label_path)
    assert data_file_name.replace('.png', '') == label_file_name.replace('.json', ''), "The data and label file name is not the same data: {} label: {}".format(data_file_name, label_file_name)

    # copy the data and label to the output path
    data_output_path = os.path.join(output_path, 'image', data_path_match +'_' + data_file_name)
    label_output_path = os.path.join(output_path, 'label_json', label_path_match +'_' + label_file_name)
    # print(data_output_path, label_output_path)

    shutil.copy(data_path, data_output_path)
    shutil.copy(label_path, label_output_path)
    
    pass

def main():
    import sys
    from pprint import pprint

    args = parse_arguments()

    # make output dir if not exist
    make_dir(args.output_path)
    print(f'[INFO] Output path: {args.output_path} is created')

    #make train, val, test dir
    make_train_val_test(args.output_path)
    print(f'[INFO] Train, Val, Test dir is created')
    
    # make image and label dir
    make_image_label(args.output_path)
    print(f'[INFO] Image and Label dir is created')

    #

    # list all the image and label
    pairs = list_all_files(args.data_path, args.label_path)
    # pprint(pairs)
    print(f'[INFO] List all the image and label')

    # split the pair into train, val, test
    train_pairs, val_pairs, test_pairs = split_data(
        args.data_path, 
        args.label_path, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )

    # copy the pair into the output dir, keeping directory structure of root path of image and label
    train_path = os.path.join(args.output_path, 'train')
    val_path = os.path.join(args.output_path, 'val')
    test_path = os.path.join(args.output_path, 'test')

    for pair in train_pairs:
        copy_and_move(pair, train_path)
    print(f'[INFO] Train data is copied')

    for pair in val_pairs:
        copy_and_move(pair, val_path)
    print(f'[INFO] Val data is copied')

    for pair in test_pairs:
        copy_and_move(pair, test_path)
    print(f'[INFO] Test data is copied')

if __name__ == '__main__':
    main()
    
    
    