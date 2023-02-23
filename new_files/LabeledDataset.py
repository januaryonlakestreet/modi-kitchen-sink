import os

import numpy as np
import pandas as pd
from torchvision.io import read_image
import random
from torch.utils.data import Dataset
import torch

class LabeledMotionDataset(Dataset):
    def __init__(self, MotionFiles, Labels):
        self.Motions = np.asarray(np.load(MotionFiles, allow_pickle=True), dtype=np.float32)
        self.MotionsTransposed = []
        for x in range(len(self.Motions)):
            self.MotionsTransposed.append(self.TransposeData(self.Motions[x]))

        self.LabelPath = Labels
        self.Labels = self.LoadLabels()

    def TransposeData(self,sample):
        sample = np.transpose(sample, (0, 2, 1))  # Transpose the data to shape [16, 64, 3]
        sample = np.expand_dims(sample, axis=2)  # Add a singleton dimension for height
        sample = np.expand_dims(sample, axis=4)  # Add a singleton dimension for width
        sample = torch.from_numpy(sample).float()  # Convert to a PyTorch tensor
        return sample

    def LoadLabels(self):
        f = open("D:\phdmethods\MoDi-main\data\Labels-pre-processing.txt", "r")
        lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.Labels)

    def __getitem__(self, idx):
        idx = random.randint(0, self.__len__())
        return self.MotionsTransposed[idx], self.Labels[idx]
