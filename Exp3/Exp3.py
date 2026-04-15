# %% [markdown]
# # EXP 3

# %% [markdown]
# ## import Libs

# %%
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F

# NumPy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# Timer
import time

# OS
import os

# ResNet18 model
from torchvision.models import resnet18

# Torchsummary for model summary
from torchsummary import summary

# %% [markdown]
# ## Device configuration as i have nvidia gpu

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# ## Hyperparameters

# %%
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# %% [markdown]
# # CIFAR-10 dataset

# %%
# CIFAR-10 dataset

# the preprocessing transformation for CIFAR-10 is required to match ResNet18 input requirements
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# ## Resnet 

# %%
# Model
resnet18 = resnet18(pretrained=False, num_classes=10).to(device)
summary(resnet18, (3, 32, 32))

# %% [markdown]
# ## make my own cnn

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = CNN().to(device)
summary(model, (3, 32, 32))

# %%
# Experiment-4: CNN with Different Configurations
# This cell sets up the framework for training CNNs with different:
# 1. Activation Functions: ReLU, Tanh, Leaky ReLU
# 2. Weight Initialization: Xavier, Kaiming, Random
# 3. Optimizers: SGD, Adam, RMSprop

import torch.nn.init as init

# Define activation functions
activation_functions = {
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.2)
}

# Weight initialization functions
def xavier_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def kaiming_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

def random_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.uniform_(module.weight, -0.1, 0.1)
        if module.bias is not None:
            init.constant_(module.bias, 0)

initialization_methods = {
    'Xavier': xavier_init,
    'Kaiming': kaiming_init,
    'Random': random_init
}

# Configurable CNN class
class ConfigurableCNN(nn.Module):
    def __init__(self, activation_fn=nn.ReLU(), num_classes=10):
        super(ConfigurableCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = activation_fn
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = activation_fn
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = activation_fn
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = self.pool(self.act3(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.act1(self.fc1(x))
        x = self.dropout(x)
        x = self.act2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Optimizer configurations
def get_optimizer(optimizer_name, model_params, lr=0.001):
    if optimizer_name == 'SGD':
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_params, lr=lr)

# Results tracking
results = {
    'activation': [],
    'initialization': [],
    'optimizer': [],
    'train_loss': [],
    'test_accuracy': []
}

print("✓ Configurations loaded successfully!")
print(f"Activation functions: {list(activation_functions.keys())}")
print(f"Initialization methods: {list(initialization_methods.keys())}")
print(f"Optimizers: ['SGD', 'Adam', 'RMSprop']")

# %%
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'cnn_cifar10.pth')

# Test the model

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# %% [markdown]
# # Resnet Test

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = resnet18(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# save the model checkpoint
torch.save(resnet18.state_dict(), 'resnet18_cifar10.pth')

# Testing loop
resnet18.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')

# %% [markdown]
# 

# %%



