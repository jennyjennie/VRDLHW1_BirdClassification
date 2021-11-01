""" 
Inference Model 
""" 

# Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

import argparse 
import numpy as np

from load_dataset import TestDataset
from config import *
from checkpoint import CheckPointStore
from init_load_model import initalize_model, load_model
from generate_anszip import write_pred_to_txt, write_ans_to_zip


# Setup hardware.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data transformation for testing data
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load test data.
data_dir = {'test': DATASET_PATH + 'testing_images'}
image_datasets = {x: TestDataset(DATASET_PATH + 'testing_img_order.txt',
                  data_dir[x], data_transforms[x]) for x in ["test"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
print(f'(Data Size) testing data: {dataset_sizes["test"]}')
testloaders = {x: DataLoader(dataset=image_datasets[x], batch_size=BATCHSIZE, 
                             pin_memory=True, num_workers=0)
              for x in ['test']}


# Test Model
def test_model(model):
    model.eval()   # Set model to evaluate mode
    print("Device : ", device)

    all_pred = []
    with torch.no_grad():
        # Iterate over data.
        for inputs, img_name in testloaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_pred.extend(predicted.cpu().tolist())

    return all_pred


def parse_args():
    parser = argparse.ArgumentParser(description='Car Classifier')
    parser.add_argument("-m", dest="model_name", default="resnet50", type=str)
    parser.add_argument("-mp", dest="model_path", type=str)
    parser.add_argument("-ftag", dest="filetag", default="", type=str)
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and test model
    global model_name
    model_name = args.model_name
    model_path = PARENT_PATH + args.model_path
    init_model = initalize_model(args.model_name, num_classes=NUM_CLASSES)
    model, checkPointStore = load_model(init_model, model_path)
    all_pred = test_model(model)

    # write prediction result to zip file
    ans_file_dir = PARENT_PATH + 'test_results'
    ansfile_name = 'answer' + model_name + '.txt'
    save_zip_name = 'answer_' + model_name + args.filetag + '.zip'
    file_ids = image_datasets["test"].get_file_ids()
    write_pred_to_txt(file_ids, all_pred, ans_file_dir, ansfile_name)
    write_ans_to_zip(ans_file_dir, ansfile_name, save_zip_name, args.filetag)


if __name__ == '__main__':
    main()
