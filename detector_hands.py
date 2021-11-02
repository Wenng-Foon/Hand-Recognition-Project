
import torch
import torch.nn as nn  
import torch.optim as optim
import torchvision.transforms as transforms 
import torchvision
import os
import cv2
import time
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import  Dataset,DataLoader

from modelresnet10 import resnet10

class Hands(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
epochs = 20

# Load Data
dataset = Hands(
    csv_file="labels.csv",
    root_dir="Data",
    transform=transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize(255),
                                  
                                       transforms.CenterCrop(224),  
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
)


train_set, test_set = torch.utils.data.random_split(dataset, [2545, 500])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = resnet10(2,3)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('starting training')
# Train Network

for epoch in range(epochs):
    losses = []
    best_testing_accuracy=0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent 
        optimizer.step()
print('training done')
def Accuracy(dataloader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
Accuracy(train_loader, model)