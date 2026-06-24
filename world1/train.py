import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nn = torch.nn


# World is 32x32, 64 dimensions should be plenty on embedding space
EMBEDDING_SIZE = 64
LOOPS = 5000
NUM_ACTIONS = 5
BATCH = 100


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


datasets = [it for it in os.listdir("training_data") if it.endswith(".npz")]
if len(datasets) == 0:
    raise ValueError(
        "Expected an npz dataset on training_data folder. Run gen_data.py."
    )

# Get just the first dataset
print("Instantiating TrainingDataset...")
dataset = TrainingDataset(f"./training_data/{datasets[0]}")
print("done")

# The purpose of this file is to create the model, and train the impact of the actions on the observations (the "world dynamics")
"""
  ┌─────────┬────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │         │           MNIST            │                               World model                           │
  ├─────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ input   │ an image x                 │ a triple (oₜ, aₜ, oₜ₊₁)                                               │
  ├─────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ model   │ a CNN                      │ an encoder f + a predictor g                                        │
  ├─────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ compute │ pred = CNN(x)              │ zₜ = f(oₜ); ẑₜ₊₁ = g(zₜ, aₜ)                                           │
  ├─────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ target  │ the label (given to you)   │ z_target = f(oₜ₊₁) — you compute it by encoding the real next frame │
  ├─────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ loss    │ cross_entropy(pred, label) │ distance(ẑₜ₊₁, z_target) — between two embeddings                   │
  └─────────┴────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

"""
# The encoder maps an observation (world state) into an embedding
encoder = nn.Sequential(
    # Input shape = [B, 1, 60, 60] <- batches of 100
    # get 16 features per 3-pixel kernel -> [16, 60, 60]
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),  # keep strong responses
    nn.MaxPool2d(2),  # halve spatial size: [16, 30, 30]
    # Same routine
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Flatten to Linear space: [32, 15, 15] -> [7200]
    nn.Flatten(),
    # Output shape = [B, EMBEDDING_SIZE] <- embedding size is chosen arbitrarily
    nn.Linear(7200, EMBEDDING_SIZE),
)
encoder.to(device)

# The target encoder will be a "delayed echo" of the main one, it will lag behind to prevent
# the optimizer from mapping every frame to the same vector, generating information collapse
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
# Copy exact weights
target_encoder.load_state_dict(encoder.state_dict())
# Freeze it
for param in target_encoder.parameters():
    param.requires_grad = False
target_encoder.to(device)
# Define momentum
m = 0.996

# The predictor maps the encoded embedding + an action into another embedding
# [B, D], action => [B, D]
# It's not a convoluted NN, but an MLP (multi-layer perceptron)
# Input: concat(z_t [B, D], onehot(a_t) [B, 5]) -> [B, D + 5]
INPUT_SIZE = EMBEDDING_SIZE + NUM_ACTIONS
predictor = nn.Sequential(
    nn.Linear(INPUT_SIZE, 128),  # 128 is arbitrary - it's a hidden layer
    nn.ReLU(),
    nn.Linear(128, EMBEDDING_SIZE),  # Output
)
predictor.to(device)

# The loss function is simply a cosine distance between the embeddings
# It will calculate how far off the prediction tensor is from the target tensor
# Options: MSELoss, F.smooth_l1_loss, cosine (1 - cos_sim)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
)

print("Training...")
# Dataloader does the batching for me
loader = DataLoader(dataset, batch_size=1024, shuffle=False)
it = 0
for before, action, after, _state in loader:
    before, action, after = before.to(device), action.to(device), after.to(device)
    obs = encoder(before)
    with torch.no_grad():
        target = target_encoder(after)
    # Now target is a fixed reference for this batch, and pred will chase it
    # Encode action into a 5-dim vector
    a_t = F.one_hot(action, num_classes=5).float()
    # Concat action vector into observation embedding
    pred_input = torch.cat((obs, a_t), dim=1)
    pred = predictor(pred_input)
    loss = loss_fn(pred, target)

    # Clear old gradients and backprop
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Hand-roll the EMA update, target will follow online encoder
    with torch.no_grad():
        for tp, p in zip(target_encoder.parameters(), encoder.parameters()):
            tp.mul_(m).add_(p, alpha=1 - m)

    # Collapse monitor -> log the spreading of the embeddings
    # avg per-dim standard deviation across the batch
    emb_spread = obs.std(dim=0).mean().item()
    # Healthy training = loss falls, spread stays meaningfully positive
    if it % 50 == 0:
        print("\r" + "" * 50, end="")
        print(f"\rloss={loss}\t\tspread={emb_spread} (it {it})")
    it += 1

print("\nTraining done (hopefully loss decreased and spread increased)")

# Now we train a single nn.Linear(64, 2) to predict (x, y) from the frozen embedding, MSE loss
probe = nn.Sequential(nn.Linear(64, 2))
probe.to(device)
p_loss_fn = nn.MSELoss()
p_optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

print("\nRunning linear probe")
new_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
it = 0
for before, _action, _after, state in new_loader:
    before, state = before.to(device), state.to(device)
    # The purpose of this probe is to understand whether the encoder has any position
    # information in it. If information collapsed, it won't (the embedding doesn't carry position in any accessible way)
    # If we can train this simple probe model to predict the position based on the encoder embeddings, we were successful
    obs_embeddings = encoder(before).detach()
    pred = probe(obs_embeddings)
    loss = p_loss_fn(pred, state.float())
    p_optim.zero_grad()
    loss.backward()
    p_optim.step()

    if it % 50 == 0:
        print(f"\rloss={loss} (it {it})")
    it += 1

print("\nFinished linear probe (loss should be close to 1 or 2)")

print("\nTest untrained encoder")
u_encoder = nn.Sequential(
    # Input shape = [B, 1, 60, 60] <- batches of 100
    # get 16 features per 3-pixel kernel -> [16, 60, 60]
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),  # keep strong responses
    nn.MaxPool2d(2),  # halve spatial size: [16, 30, 30]
    # Same routine
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Flatten to Linear space: [32, 15, 15] -> [7200]
    nn.Flatten(),
    # Output shape = [B, EMBEDDING_SIZE] <- embedding size is chosen arbitrarily
    nn.Linear(7200, EMBEDDING_SIZE),
)
u_encoder.to(device)
it = 0
print("\nRunning untrained encoder")
# A randomly initialized, untrained encoder should still preserve spatial structure fairly well
u_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
for before, _action, _after, state in u_loader:
    before, state = before.to(device), state.to(device)
    # untrained encoder
    obs_embeddings = u_encoder(before).detach()
    pred = probe(obs_embeddings)
    loss = p_loss_fn(pred, state.float())
    p_optim.zero_grad()
    loss.backward()
    p_optim.step()

    if it % 50 == 0:
        print(f"\rloss={loss} (it {it})")
    it += 1

print("\nFinished (untrained encoder should have loss ~8.25)")
