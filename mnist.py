import gzip
import matplotlib.pyplot as plt
import sys
print(sys.version)
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import pandas as pd

#print(np.get_printoptions())
np.set_printoptions(threshold = 100000)

#Hyperparameters:
batch_size = 200


#open test image data:
with gzip.open('data/DigitsMNIST/train-images-idx3-ubyte.gz', 'rb') as f:
    training_data = f.read()

with gzip.open('data/DigitsMNIST/train-labels-idx1-ubyte.gz') as f:
	training_labels = f.read()

with gzip.open('data/DigitsMNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_data = f.read()

with gzip.open('data/DigitsMNIST/t10k-labels-idx1-ubyte.gz') as f:
	test_labels = f.read()


#non-image data at beginning of training data file
magic_num = int.from_bytes(training_data[0:4], "big")
len_data = int.from_bytes(training_data[4:8], "big")
rows = int.from_bytes(training_data[8:12], "big")
cols = int.from_bytes(training_data[12:16], "big")
print("Magic number:", magic_num)
print("Length of data:", len_data)
print("Pixel rows:", rows)
print("Pixel cols:", cols)

#non-label data at beginning of training label file:
label_magic_num = int.from_bytes(training_labels[0:4], "big")
label_len_data = int.from_bytes(training_labels[4:8], "big")
print("Label magic number:", label_magic_num)
print("Length of label data:", label_len_data)

test_magic_num = int.from_bytes(test_data[0:4], "big")
test_len_data = int.from_bytes(test_data[4:8], "big")
test_rows = int.from_bytes(test_data[8:12], "big")
test_cols = int.from_bytes(test_data[12:16], "big")
print("Magic number, test:", test_magic_num)
print("Length of test data:", test_len_data)
print("Test pixel rows:", test_rows)
print("Test pixel cols:", test_cols)

#non-label data at beginning of training label file:
test_label_magic_num = int.from_bytes(test_labels[0:4], "big")
test_label_len_data = int.from_bytes(test_labels[4:8], "big")
print("Test label magic number:", test_label_magic_num)
print("Test length of label data:", test_label_len_data)

# practice showing an image
# first_image = np.zeros((28,28))
# current_index=16 #image data starts here
# for i in range(28):
# 	for j in range(28):
# 		first_image[i,j]=training_data[current_index]
# 		current_index = current_index + 1

# #plt.matshow(first_image, cmap="gray")
# plt.show()

#Save each image's data into a separate .csv file:
#training data set
if not (os.path.exists("data/DigitsMNIST/img_dir")):
	os.makedirs("data/DigitsMNIST/img_dir")

if not (os.path.exists("data/DigitsMNIST/img_dir/digit59999.csv")):
	for i in range(len_data):
		if i%1000 == 0:
			print(i)
		new_image=np.zeros(784)
		for j in range(784):
			new_image[j]=training_data[16 + 784*i + j]
		new_file_name = "data/DigitsMNIST/img_dir/digit" + str(i) + ".csv"
		new_image = new_image.astype(int)
		pd.DataFrame(new_image).to_csv(new_file_name, index=False)

#test data set
if not (os.path.exists("data/DigitsMNIST/img_dir_test")):
	os.makedirs("data/DigitsMNIST/img_dir_test")

if not (os.path.exists("data/DigitsMNIST/img_dir_test/digit9999.csv")):
	for i in range(test_len_data):
		if i%1000 == 0:
			print(i)
		new_image=np.zeros(784)
		for j in range(784):
			new_image[j]=test_data[16 + 784*i + j]
		new_file_name = "data/DigitsMNIST/img_dir_test/digit" + str(i) + ".csv"
		new_image = new_image.astype(int)
		pd.DataFrame(new_image).to_csv(new_file_name, index=False)

#Save training labels into a .csv
#print(len_data)
if not (os.path.exists("data/DigitsMNIST/training_labels.csv")):
	labels_np = np.zeros(len_data)
	file_names = np.empty(len_data, dtype='object')
	for i in range(len_data):
		file_names[i] = "digit" + str(i) + ".csv"
		labels_np[i] = training_labels[8+i]
	labels_np = labels_np.astype(int)
	d = {"files": file_names, "labels": labels_np}
	df = pd.DataFrame(d)
	df.to_csv("data/DigitsMNIST/training_labels.csv", index=False)

#Save test labels into a .csv file
if not (os.path.exists("data/DigitsMNIST/test_labels.csv")):
	labels_np = np.zeros(test_len_data)
	file_names = np.empty(test_len_data, dtype='object')
	for i in range(test_len_data):
		file_names[i] = "digit" + str(i) + ".csv"
		labels_np[i] = test_labels[8+i]
	labels_np = labels_np.astype(int)
	d = {"files": file_names, "labels": labels_np}
	df = pd.DataFrame(d)
	df.to_csv("data/DigitsMNIST/test_labels.csv", index=False)

#Now we build our dataset:
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_data = CustomImageDataset("data/DigitsMNIST/training_labels.csv", "data/DigitsMNIST/img_dir", transform=transforms.ToTensor())
test_data = CustomImageDataset("data/DigitsMNIST/test_labels.csv", "data/DigitsMNIST/img_dir_test", transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle=True)