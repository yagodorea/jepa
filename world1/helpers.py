import os
import torch

from consts import EMBEDDING_SIZE, HIDDEN_LAYERS, NUM_ACTIONS
from training_dataset import TrainingDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nn = torch.nn


def load_dataset():
    datasets = [it for it in os.listdir("training_data") if it.endswith(".npz")]
    if len(datasets) == 0:
        raise ValueError(
            "Expected an npz dataset on training_data folder. Run gen_data.py."
        )

    # Get just the first dataset
    print("Instantiating TrainingDataset...")
    dataset = TrainingDataset(f"./training_data/{datasets[0]}")
    return dataset


def load_model():
    print("Loading model...")
    ckpt = torch.load("model.pt", map_location=device)
    encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7200, EMBEDDING_SIZE),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    INPUT_SIZE = EMBEDDING_SIZE + NUM_ACTIONS
    predictor = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_LAYERS),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYERS, EMBEDDING_SIZE),  # Output
    )
    predictor.to(device)
    predictor.load_state_dict(ckpt["predictor"])
    # Same as encoder
    target_encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7200, EMBEDDING_SIZE),
    )
    target_encoder.to(device)
    target_encoder.load_state_dict(ckpt["target_encoder"])
    return (
        encoder,
        predictor,
        target_encoder,
    )
