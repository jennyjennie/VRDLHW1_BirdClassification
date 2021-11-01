"""
Initialize model and load model function
"""

# Imports
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet # DBG : cannot inference
from bottleneck_transformer_pytorch import BottleStack

import numpy as np

from config import * 
from checkpoint import CheckPointStore


# Setup hardware.
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
    device = "cuda:0"
else:
    map_location = "cpu"
    device = "cpu"


# Initialize model.
def initalize_model(pretrain_model_name, num_classes):
    """
     Initialize pretrained model
    """
    print(f'pretrain_model_name: {pretrain_model_name}')

    if pretrain_model_name == "resnet50":
        """ ResNet-50 """
        model_init = models.resnet50(pretrained=True) # Initialize the pretrained model
        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)  # replace final fully connected layer

    elif pretrain_model_name == "resnet101":
        """ ResNet-101 """
        model_init = models.resnet101(pretrained=True)

        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)  

    elif pretrain_model_name == "densenet121":
        """ Desnet-121 """
        model_init = models.densenet121(pretrained=True) 
        num_ftrs = model_init.classifier.in_features
        model_init.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif pretrain_model_name == "resnext50_32x4d":
        """ ResNeXt-50-32x4d """
        model_init = models.resnext50_32x4d(pretrained=True)
        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)     
    
    elif pretrain_model_name == "resnext101_32x8d":
        """ resnext101_32x8d """
        model_init = models.resnext101_32x8d(pretrained=True)
        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)  
        
    return model_init


def load_model(model, model_path = None,  IS_TRAINING = False, IS_RESUME_TRAINIG = False):
    checkPointStore = CheckPointStore()

    # load trained model if needed. 
    if ((IS_TRAINING and model_path is not None) or not IS_TRAINING):
        print("Load trained model...")
       # checkpoint = torch.load(CHECKPOINT_PATH)
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])

        # save checkpoint info in object
        checkPointStore.total_epoch = checkpoint['total_epoch']
        checkPointStore.Best_val_Acc = checkpoint['Best_val_Acc']
        checkPointStore.epoch_loss_history = checkpoint['epoch_loss_history']
        checkPointStore.epoch_acc_history = checkpoint['epoch_acc_history']
        checkPointStore.model_state_dict = checkpoint['model_state_dict']
        print(f'Trained epochs previously : {checkPointStore.total_epoch}\nBest val Acc : {checkPointStore.Best_val_Acc}')
        print("Successfully loaded trained model.")
    else:
        print("No trained model found, using built-in pretrained model.")

    print(f'Send model to {device}')
    model.to(device)
    
    return model, checkPointStore
