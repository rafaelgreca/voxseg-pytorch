from typing import Dict
from torch.utils.data import Dataset


class AVA_Dataset(Dataset):
    """
    Creates a torch Dataset to be used in the training step.
    """

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index) -> Dict:
        return {"X": self.X[index], "y": self.y[index]}
