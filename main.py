import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import random
import network
from pycm import ConfusionMatrix


EPOCHS_COUNT = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.00005
TOTAL_IDENTITIES = 10177
CELEBA_DATASET_PATH = "~/Datasets/PyTorch"


def get_identities_tensor(dataset_item):
	identity = int(dataset_item)
	identities_list = [-1] * TOTAL_IDENTITIES
	identities_list[identity - 1] = 1
	identities_tensor = torch.tensor(identities_list)
	return identities_list


def get_batch_identities_tensor(dataset_items_batch):
	batch_identities_list = []
	for item in dataset_items_batch:
		batch_identities_list.append(get_identities_tensor(item))
	batch_identities_tensor = torch.tensor(batch_identities_list)
	return batch_identities_tensor


def initialize_network():
	global device
	global model
	global optimizer
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = network.Network().to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def initialize_dataset(dataset_path=CELEBA_DATASET_PATH):
	global dataset
	global train_loader
	global validation_loader
	dataset = torchvision.datasets.CelebA(dataset_path, target_type=["identity"], transform=torchvision.transforms.ToTensor())
	# dataset = [full_dataset[i] for i in range(25000)]
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)


def validate():
	global input_labels
	global predicted_labels
	input_labels = []
	predicted_labels = []
	model.eval()
	total_true_positives = 0
	validation_dataset_size = 0
	with torch.no_grad():
		for batch_index, (data, target) in enumerate(validation_loader):
			if batch_index >= 10000:
				break
			data, target = data.to(device), int(target.to(device)) - 1
			network_output = model(data)[0]
			network_output_index = int(network_output.argmax())
			if network_output_index == target:
				total_true_positives += 1
			input_labels.append(target)
			predicted_labels.append(network_output_index)
			validation_dataset_size += 1
	accuracy = total_true_positives / validation_dataset_size
	print("Accuracy:", accuracy)


def print_confusion_matrix(input_labels, predicted_labels):
	cm = ConfusionMatrix(input_labels, predicted_labels)
	# print(cm)


def train_all(epochs_count=EPOCHS_COUNT, batch_size=BATCH_SIZE):
	model.train()
	for i in range(epochs_count):
		training_loss = 0
		batches_count = 0
		print("Epoch number", i + 1, "start")
		for batch_index, (data, target) in enumerate(train_loader):
			if batch_index % 10 == 0:
				print("batch_index:", batch_index)
			data, target = data.to(device), target.to(device)
			network_output = model(data)
			optimizer.zero_grad()
			loss_value = network.loss(network_output.to(device), get_batch_identities_tensor(target).to(device))
			training_loss += loss_value.item()
			loss_value.backward()
			optimizer.step()
			batches_count += 1
		training_loss /= batches_count
		print("Epoch number", i + 1, "end")
		print("Training loss:", training_loss)
		validate()
		print("---------------------")


if __name__ == "__main__":
	initialize_dataset()
	initialize_network()
	train_all()
	print_confusion_matrix(input_labels, predicted_labels)
