import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from evaluation.models.stgcn import STGCN
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file, allow_pickle=True)
        #self.data = self.data[:, :15]
        #self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = label_file
        print("r")
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
    EpochCorrect = 0
    global TotalCorrect
    global TotalEvaluated
    TestData = train_dataset  # 0 = data 1 = labels
    RandomSampleCount = 32
    AccuracyLoader = DataLoader(TestData, batch_size=RandomSampleCount, shuffle=True)
    AllConstructedBatches = []

    for i, batch in enumerate(AccuracyLoader):
        ConstructedBatch = \
            {
                'x': batch[0],
                'yhat': batch[1],
                'y': batch[1]
            }
        AllConstructedBatches.append(convert_to_tensor(ConstructedBatch))
    if len(AllConstructedBatches) != 0:
        SelectedBatch = AllConstructedBatches[random.randint(0, len(AllConstructedBatches)-1)]
        OutputValues = model(SelectedBatch)
        yhat = OutputValues["yhat"].max(dim=1).indices
        ygt = OutputValues["y"]

        for x in range(len(yhat)):
            TotalEvaluated += 1
            if yhat[x] == ygt[x]:
                EpochCorrect += 1
        TotalCorrect += EpochCorrect
    else:
        return "no batches"
    return "epoch correct ", EpochCorrect, "epoch percentage ", EpochCorrect / RandomSampleCount, "average correct ", TotalCorrect / TotalEvaluated


# Define hyperparameters
batch_size = 32
lr = 1e-3
num_epochs = 250

# run time
TotalCorrect = 0
TotalEvaluated = 0
# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = STGCN(in_channels=3, num_class=876, graph_args={"layout": 'openpose', "strategy": "spatial"},
              edge_importance_weighting=True, device=device)

model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load dataset
train_dataset = CustomDataset('../data/edge_rot_joints.npy', label_to_integer(LoadLabels()))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Train model
def convert_to_tensor(batch_dict):
    tensor_dict = {}
    for key, value in batch_dict.items():
        tensor_dict[key] = torch.tensor(value, dtype=torch.float32).to(device)
    return tensor_dict


for epoch in range(num_epochs):

    epoch_loss = 0.0
    epoch_acc = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # batch = batch.to('cuda')

        ConstructedBatch = \
            {
                'x': batch[0],
                'yhat': batch[1],
                'y': batch[1]
            }

        outputs = model(convert_to_tensor(ConstructedBatch))

        loss = model.criterion(outputs["yhat"], convert_to_tensor(ConstructedBatch)["y"].to(torch.long))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print("Epoch ", epoch + 1,  "average loss ", epoch_loss / len(train_loader), " ",   EvaluateAccuracy())


from datetime import date

today = date.today()
torch.save(model.state_dict(), str(today) + 'model.pt')
