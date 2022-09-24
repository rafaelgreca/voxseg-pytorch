import pandas as pd
import numpy as np
from typing import Dict
from torch.utils.data import Dataset


class Preprocessing_Dataset(Dataset):
    """
    Creates a torch Dataset to be used in the preprocessing step.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        return {
            "recording-id": self.df["recording-id"][index],
            "extended filename": self.df["extended filename"][index],
            "start": self.df["start"][index],
            "label": self.df["label"][index],
        }


def custom_collate(data: Dict):
    return data


class AVA_Dataset(Dataset):
    """
    Creates a torch Dataset to be used in the training step.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict:
        return {"X": self.X[index], "y": self.y[index]}
