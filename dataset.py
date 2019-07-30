import torch
import numpy
from torch.utils.data.dataset import Dataset
from data import Data

class PromiseDataset(Dataset):
    def __init__(self, is_train = True):
        self.is_train = is_train
        if self.is_train:
            self.data, self.label = Data().load_train_data("promise12/train")
        else:
            self.data, self.label = Data().load_test_data("promise12/test")

    def __getitem__(self, index):
        if isinstance(self.data, numpy.ndarray):
            self.data = torch.from_numpy(self.data)

        if isinstance(self.label, numpy.ndarray):
            if self.is_train:
                self.label = torch.from_numpy(self.label)

        if self.is_train:
            return self.data[index].float(), self.label[index].float()
        else:
            return self.data[index].float(), self.label[index]

    def __len__(self):

        return self.data.shape[0]
