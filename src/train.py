"""
Train Model
"""

# Imports
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import copy
import time
import argparse
import pandas as pd

from config import *
from load_dataset import BirdDataSet
from init_load_model import initalize_model, load_model

torch.manual_seed(42)


# Setup hardware.
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
    device = "cuda:0"
else:
    map_location = "cpu"
    device = "cpu"


# Data augmentation and normalization for training
transform_options = [
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomRotation(degrees=[-15, 15]),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomAffine(0, shear=20),
]

# Data transformation for training and validation.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
             transforms.RandomChoice(transform_options)
        ], p=0.9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Load the training dataset, split to training set(80/100)
# and validation set(20/100).
dataset = pd.read_csv(DATASET_PATH+'new_training_labels.txt',
                      sep=' ', names=['id', 'label_int'])
train_info, val_info = train_test_split(dataset, test_size=0.2,
                                        stratify=dataset['label_int'])
data_dir = {'train': DATASET_PATH+'training_images',
            'val': DATASET_PATH+'training_images'}
info_file = {'train': train_info, 'val': val_info}
image_datasets = {x: BirdDataSet(info_file[x], data_dir[x],
                  transform=data_transforms[x]) for x in ['train', 'val']}
weightedRandomSampler = image_datasets["train"].getWeightedRandomSampler()
dataloaders = {}
dataloaders['train'] = DataLoader(dataset=image_datasets['train'],
                                  batch_size=BATCHSIZE,
                                  sampler=weightedRandomSampler,
                                  pin_memory=True)
dataloaders['val'] = DataLoader(dataset=image_datasets['val'],
                                batch_size=BATCHSIZE,
                                shuffle=True,
                                pin_memory=True)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(f'(Data Size) Training data: {dataset_sizes["train"]}, \
                    Valiation data: {dataset_sizes["val"]}')


def write_training_info_txtfile(phase, epochs, trained_epochs,
                                num_epochs, epoch_loss, running_corrects,
                                epoch_acc, filetag):
    '''Write training and validation information into text file'''
    train_file_dir = 'train_results/training_results_'
    file_path = PARENT_PATH + train_file_dir + filetag + '.txt'
    file = open(file_path, 'a+')

    if phase == 'train':
        file.write('\n')
        file.write('Epoch {}/{}\n'.format(epochs, num_epochs - 1))
        file.write('-'*10+'\n')
        file.write('{} Loss: {:.4f} Acc: {}/{}({:.4f})\n'.format(
                phase, epoch_loss, running_corrects,
                dataset_sizes[phase], epoch_acc))
        file.write('\n')
    else:
        file.write('{} Loss: {:.4f} Acc: {}/{}({:.4f})\n'.format(
                phase, epoch_loss, running_corrects,
                dataset_sizes[phase], epoch_acc))
        file.write('\n')
        file.write('total_epoch: {}\n'.format(trained_epochs + epochs))
        file.write('Best_val_Acc: {}\n'.format(epoch_acc))

    file.close()


def create_model_save_file(IS_RESUME_TRAINIG):
    '''Determine where the model stored.'''  # DBG : To be modified
    contents = os.listdir(MODEL_SAVE_ROOT_PATH)
    try:
        idx = contents.index('.DS_Store')
        del contents[idx]
    except:
        pass

    if os.listdir(MODEL_SAVE_ROOT_PATH):
        directory_contents = sorted(contents, key=lambda list_dir: int(list_dir[3:]))
    else:
        directory_contents = []

    if len(directory_contents) == 0:
        model_filename = 'exp1'
    elif (IS_RESUME_TRAINIG):
        model_filename = 'exp' + str(int(directory_contents[-1][3:]))
    else:
        model_filename = 'exp' + str(int(directory_contents[-1][3:])+1)

    return model_filename


# train model
def train_model(model, criterion, optimizer, scheduler,
                checkPointStore, IS_RESUME_TRAINIG, num_epochs=25):
    print(f'Training on Device : {device}')

    model_filename = create_model_save_file(IS_RESUME_TRAINIG)
    tag = model_filename
    # Save model to the corresponding file
    model_save_path = MODEL_SAVE_ROOT_PATH + '/' + model_filename + '/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    trained_epochs = 0
    best_acc = 0.0
    epoch_loss_history = {x: list() for x in ["train", "val"]}  # record the loss in each epoch
    epoch_acc_history = {x: list() for x in ["train", "val"]}  # record  the accuracy in each epoch

    # Reloaded trained infomation if resume training.
    if (IS_RESUME_TRAINIG):
        trained_epochs = checkPointStore.total_epoch
        best_acc = checkPointStore.Best_val_Acc
        epoch_loss_history = checkPointStore.epoch_loss_history
        epoch_acc_history = checkPointStore.epoch_acc_history

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss_history[phase].append(epoch_loss)
            epoch_acc_history[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {}/{}({:.4f})'.format(
                phase, epoch_loss, running_corrects, dataset_sizes[phase], epoch_acc))

            # Save the model if the model has the best validation acc.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                            'total_epoch': trained_epochs + epoch,
                            'Best_val_Acc': best_acc,
                            'model_state_dict': model.state_dict(),
                            'epoch_loss_history': epoch_loss_history,
                            'epoch_acc_history': epoch_acc_history
                           }, model_save_path + MODEL_SAVE_NAME + ".pth")  # save on GPU mode
                print("Save model info based on val acc.")
                write_training_info_txtfile(phase, epoch, trained_epochs, num_epochs,
                                            epoch_loss, running_corrects, best_acc, tag)
            # Save the model every five epochs.
            elif (epoch + 1) % 5 == 0:
                # save model when final epoch and not accroding validation accuracy
                torch.save({
                            'total_epoch': trained_epochs + epoch,
                            'Best_val_Acc': epoch_acc,
                            'model_state_dict': model.state_dict(),
                            'epoch_loss_history': epoch_loss_history,
                            'epoch_acc_history': epoch_acc_history
                           }, model_save_path + MODEL_SAVE_NAME + '_e' + str(int(epoch+1)) + ".pth")
                print("End final epoch - Save model info.")
                write_training_info_txtfile(phase, epoch, trained_epochs, num_epochs,
                                            epoch_loss, running_corrects, epoch_acc, tag)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(10)


def parse_args():
    parser = argparse.ArgumentParser(description='Bird Classifier')
    parser.add_argument("-m", dest="model_name", default="resnet50", type=str)
    parser.add_argument("-e", dest="num_epoch", default=NUM_EPOCHS, type=int)
    parser.add_argument("-rmp", dest="model_name_resume_path", default=None, type=str)
    parser.add_argument("-lr", dest="learning_rate", default=0.001, type=float)
    parser.add_argument("-ftag", dest="filetag", default='', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    global MODEL_SAVE_NAME
    MODEL_SAVE_NAME = args.model_name + args.filetag

    if args.model_name_resume_path is not None:
        model_name_resume_path = MODEL_SAVE_ROOT_PATH + args.model_name_resume_path
    else:
        model_name_resume_path = None

    # Show a random batch of images.
    inputs, labels = next(iter(dataloaders['train']))  # Get a batch of training data
    out = torchvision.utils.make_grid(inputs)  # Make a grid from batch
    imshow(out, title=labels)

    init_model = initalize_model(args.model_name, NUM_CLASSES)  # initalize built-in pretrained model
    model, checkPointStore = load_model(init_model,
                                        model_path=model_name_resume_path,
                                        IS_TRAINING=True)  # load trained model and trained info in CheckPointStore object
    print('model used: ', model)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.learning_rate)

    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=4)

    IS_RESUME_TRAINIG = False if args.model_name_resume_path is None else True

    model_ft = train_model(model, criterion, optimizer_ft,
                           exp_lr_scheduler, checkPointStore,
                           IS_RESUME_TRAINIG, num_epochs=args.num_epoch)


if __name__ == '__main__':
    main()
