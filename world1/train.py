import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from consts import EMBEDDING_SIZE, HIDDEN_LAYERS, MODEL_LR, NUM_ACTIONS
from helpers import load_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nn = torch.nn

dataset = load_dataset()

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
    nn.Linear(INPUT_SIZE, HIDDEN_LAYERS),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYERS, EMBEDDING_SIZE),  # Output
)
predictor.to(device)

# The loss function is simply a cosine distance between the embeddings
# It will calculate how far off the prediction tensor is from the target tensor
# Options: MSELoss, F.smooth_l1_loss, cosine (1 - cos_sim)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()), lr=MODEL_LR
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

print(
    "\nTraining done (hopefully loss decreased for a bit then slowly increased, and spread increased)"
)

print("Saving model...")

torch.save(
    {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
    },
    "model.pt",
)

print("Finished")
