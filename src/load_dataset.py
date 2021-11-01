"""
Load Custom Dataset (Bird Dataset)
"""

# Imports
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import numpy as np
from config import *
from sklearn.model_selection import train_test_split


# Load custom data set. In this case, load the bird data set of VRDL HW1.
class BirdDataSet(Dataset):
	def __init__(self, info_file, root_dir, transform=None):
		# The info_file is expected to be a .txt
		#self.annotations = pd.read_csv(info_file, sep = ' ', names = ['id', 'label_int'])
		# The info file has been transformed to pd df for train_test_split.
		self.annotations = info_file
		self.file_ids = self.annotations['id']
		self.labels = self.annotations['label_int']
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
		image = Image.open(img_path).convert("RGB")
		ylabel = torch.tensor(self.annotations.iloc[index, 1])
		if self.transform:
			image = self.transform(image)
		return (image, ylabel)

	def getWeightedRandomSampler(self):
		labels = np.array(self.labels, dtype="int64")
		class_sample_count = np.zeros(NUM_CLASSES)
		for i in range(NUM_CLASSES):
			class_sample_count[i]  = sum(labels == i)

		weight = 1. / class_sample_count
		samples_weight = torch.tensor([weight[t] for t in labels])
		sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

		return sampler

class TestDataset(Dataset):
	def __init__(self, testing_order_file, data_dir, transform=None):
		self.annotations = pd.read_csv(testing_order_file, names = ['id'])
		self.file_ids = self.annotations['id']
		self.data_dir = data_dir
		self.transform = transform

	def __getitem__(self, idx):
		img_path = self.data_dir + "/" + self.file_ids[idx]
		img = Image.open(img_path).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)

		return img, self.file_ids[idx]

	def __len__(self):
		return len(self.file_ids)

	def get_file_ids(self):
		return self.file_ids
