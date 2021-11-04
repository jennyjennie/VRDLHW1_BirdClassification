"""
 Change classes.txt to dict which can map num to class and vice versa
 And setup some configurations
"""

# Imports
from pathlib import Path
import os


# Configs
IMG_SIZE = 224

BATCHSIZE = 32

NUM_CHANNEL = 3  # RGB img

NUM_CLASSES = 200  # The number of classes

NUM_EPOCHS = 300

PARENT_PATH = str(Path(os.getcwd()).parent.absolute()) + '/'

CHECKPOINT_PATH = ""

DATASET_PATH = PARENT_PATH + '2021VRDL_HW1_datasets/'

CLASS_FILENANE = DATASET_PATH + 'classes.txt'

TRAINING_LABEL_FILENAME = DATASET_PATH + 'training_labels.txt'

MODEL_SAVE_ROOT_PATH = PARENT_PATH + 'model/'


# Convert classes to int
def ConvertFileToDict(filename=CLASS_FILENANE):
    LABELS = {}
    LABELS_TO_INT = {}
    file = open(filename, 'r')
    for line in file.readlines():
        num, breed = line.strip().split('.')
        num = int(num) - 1
        LABELS[num] = breed
        LABELS_TO_INT[breed] = num
    file.close()

    return LABELS, LABELS_TO_INT


# Create a new training label for training, validation and testing.
def NewTrainingLabels(filename=TRAINING_LABEL_FILENAME):
    file = open(filename, 'r')
    newfile = open(DATASET_PATH + 'new_training_labels.txt', 'w')
    for line in file.readlines():
        name, breed = line.strip().split(' ')
        breed_num = breed[:3]
        # The class has to be from 0 ~ 199.
        line = name + ' ' + str(int(breed_num) - 1)
        newfile.write(line + '\n')
    file.close()
    newfile.close()


# Create dicts for labels to int and int to lable.
LABELS, LABELS_TO_INT = ConvertFileToDict()

# Create new training labels.txt.
NewTrainingLabels()
