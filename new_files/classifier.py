import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(16, 32, kernel_size=(4, 4, 4), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.Dropout1 = nn.Dropout(0.9)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.Dropout2 = nn.Dropout(0.9)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.Dropout3 = nn.Dropout(0.9)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.Dropout4 = nn.Dropout(0.9)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.Dropout5 = nn.Dropout(0.9)

        self.conv6 = nn.Conv3d(512, 1024, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.Dropout6 = nn.Dropout(0.9)

        self.fc1 = nn.Linear(1024 * 1 * 1 * 2, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 876)

        self.LeakyReLU = nn.LeakyReLU(0.9)
        self.ELU = nn.ELU(0.22)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.ELU(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.Dropout1(x)
        print(x.shape)

        x = self.conv2(x)
        print(x.shape)
        x = self.ELU(x)
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = self.Dropout2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.ELU(x)
        x = self.pool3(x)
        x = self.Dropout3(x)

        x = self.conv4(x)
        x = self.ELU(x)
        x = self.pool4(x)
        x = self.Dropout4(x)




        x = x.view(16, -1)
        x = self.ELU(self.fc1(x))
        x = self.ELU(self.fc2(x))

        x = self.fc3(x)

        return x


# Define a custom dataset to load your data
class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file)
        self.labels = label_file

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def LoadLabels():
    f = open("D:\phdmethods\MoDi-main\data\Labels-Processed.txt", "r")
    lines = f.readlines()
    return lines


def label_to_integer(labels):
    unique_labels = list(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    return [label_map[label] for label in labels]


def EvaluateAccuracy():
    global TotalCorrect
    global TotalEvaluated
    TestData = dataset  # 0 = data 1 = labels
    RandomSampleCount = 16
    TestMotion = []
    TestLabels = []
    for x in range(RandomSampleCount):
        TestMotion.append(TestData[x][0])
        TestLabels.append(TestData[x][1])

    outputValue = model(torch.tensor(np.asarray(TestMotion)).float())

    predictedClass = outputValue.max(dim=1).indices
    for x in range(len(predictedClass)):
        TotalEvaluated += 1
        if predictedClass[x] == TestLabels[x]:
            TotalCorrect += 1

    return "Total correct ", TotalCorrect, " out of a total guesses ", TotalEvaluated, " ", TotalCorrect / TotalEvaluated


# Define hyperparameters and instantiate the model and dataset
batch_size = 16
learning_rate = 0.00000005
num_epochs = 5000

TotalCorrect = 0
TotalEvaluated = 0

model = Net()
dataset = CustomDataset('../data/motion.npy', label_to_integer(LoadLabels()))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
        inputs, labels = data
        if batch_size == len(inputs):
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss / len(dataset)))
    print(EvaluateAccuracy())
# Save the model
torch.save(model.state_dict(), 'model.pt')
