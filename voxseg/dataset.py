from torch.utils.data import Dataset


class AVA_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {"X": self.X[index], "y": self.y[index]}
