import torch
from torch.utils.data import Dataset


class RockData(Dataset):

    def __init__(self, X, y, need_deal=False, merge_label=True, train=True):
        self.X = X
        self.y = y
        self.need_deal, self.merge_label, self.train = need_deal, merge_label, train
    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()
        return X, y
    
    def __len__(self):
        return len(self.X)

if __name__ == '__main__':
    label = torch.tensor(1).long()
    label = 3
    print(label)