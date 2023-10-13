import torch
from torch.utils.data import Sampler


class WeightDecaySampler(Sampler):
    def __init__(self, weight):
        self.weight = torch.tensor(weight, dtype=torch.double)

    def __iter__(self):
        print("===> weight without decay:")
        print(self.weight)
        index = torch.multinomial(self.weight, len(self.weight), replacement=True)
        self.weight[index] = self.weight[index] * 0.96
        print("===> weight after decay:")
        print(self.weight)
        return iter(index)

    def __len__(self):
        return len(self.weight)