import numpy as np
import torch
from torch.utils.data import Dataset

from consts import BATCH


class TrainingDataset(Dataset):
    def __init__(self, npz_path) -> None:
        self.data_archive = np.load(npz_path)
        print("Extracting observations")
        self.observations = self.data_archive["observations"]
        print("Extracting actions")
        self.actions = self.data_archive["actions"]
        print("Extracting states")
        self.states = self.data_archive["states"]
        self.valid_starts = [
            i for i in range(len(self.actions)) if (i + 1) % BATCH != 0
        ]

    def __len__(self):
        return len(self.valid_starts)

    def index(self, i: int) -> int:
        return self.valid_starts[i]

    def __getitem__(self, index):
        i = self.index(index)
        ob_t = torch.as_tensor(self.observations[i])
        ac_t = torch.as_tensor(self.actions[i])
        # print(f"ob_t.shape: {ob_t.shape}")
        ob_tp1 = torch.as_tensor(self.observations[i + 1])
        st_t = torch.as_tensor(self.states[i])
        return ob_t, ac_t, ob_tp1, st_t
