# Change classes.txt to dict which can map num to class and vice versa

IMG_SIZE = 224 

BATCHSIZE = 4

NUM_CHANNEL = 3 # RGB img

NUM_CLASSES = 200 # The number of classes

NUM_EPOCHS = 50

NUM_FREEZE_LAYERS = 0

MODEL_SAVE_ROOT_PATH = "model/"

CHECKPOINT_PATH = ""

IS_RESUME_TRAINIG = False

IS_SELECT_MODEL_BY_VAL_ACC = True


CLASS_FILENANE = 'classes.txt'
TRAINING_LABEL_FILENAME = 'training_labels.txt'


def ConvertFileToDict(filename = CLASS_FILENANE):
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

def NewTrainingLabels(filename = TRAINING_LABEL_FILENAME):
	file = open(filename, 'r')
	newfile = open('new_training_labels.txt', 'w')
	for line in file.readlines():
		name, breed = line.strip().split(' ')
		breed_num = breed[:3]
		# The class has to be from 0 ~ 199.
		line = name + ' ' + str(int(breed_num) - 1)
		newfile.write(line + '\n')
	file.close()
	newfile.close()

LABELS, LABELS_TO_INT = ConvertFileToDict()
NewTrainingLabels()
#print(LABELS_TO_INT)

