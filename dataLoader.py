class DataLoader():
	"""
	Class for loading image data stored in a zip folder into the system
	Input: path of the zipfile
	Output: Train and Test loader with classes of the data
	"""
	def __init__(self, zip_path):
		self.zip_path = zip_path

	def zip_extractor(self):
		"""Extract images from zipfile and create data directory and get train & test folders"""
		import os
		import zipfile
		lst1 = os.listdir()
		zip_ref = zipfile.ZipFile(self.zip_path, 'r')
		zip_ref.extractall(os.getcwd())
		zip_ref.close()
		lst2 = os.listdir()
		self.data_dir = os.path.join(os.getcwd(), list(set(lst2) - set(lst1))[0])
		self.trainFolder = os.path.join(self.data_dir, 'Train')
		self.testFolder = os.path.join(self.data_dir, 'Test')


	def dataloader(self):
		"""
		Load and converts the data in pytorch tensor format for neural network
		Output: train & test data loader and classes of data
		"""
		import os
		import numpy as np
		import torch, torchvision
		# call zip_extractor
		self.zip_extractor()
		# tranform imnage data
		transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                            torchvision.transforms.ToTensor()])
		# load train datasets
		train_dataset = torchvision.datasets.ImageFolder(root=self.trainFolder, transform=transform)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=20, num_workers=0, shuffle=True)
		# load test datasets
		test_dataset = torchvision.datasets.ImageFolder(root=self.testFolder, transform=transform)
		test_loader = torch.utils.data.DataLoader(train_dataset,batch_size=20,num_workers=0,shuffle=True)
		# get classes of the data
		classes = os.listdir(self.trainFolder)

		return train_loader, test_loader, classes

	def train_batch_visualizer(self):
		"""visualise a batch of training data"""
		import numpy as np
		import torch, torchvision
		import matplotlib.pyplot as plt
		# obtain one batch of training images
		train_loader, _, classes = self.dataloader()
		dataiter = iter(train_loader)
		images, labels = dataiter.next()
		images = images.numpy()
		# plot the images in the batch, along with the corresponding labels
		fig = plt.figure(figsize=(25, 4))
		for idx in np.arange(20):
		    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
		    ax.imshow(np.squeeze(images[idx]), cmap='gray')
		    # print out the correct label for each image
		    ax.set_title(classes[labels[idx]])
		plt.show()

		# remove created directory from zipfile
		from shutil import rmtree
		rmtree(self.data_dir)


	