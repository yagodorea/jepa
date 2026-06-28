import torch
from torch.utils.data import DataLoader

from consts import EMBEDDING_SIZE
from helpers import load_model, load_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nn = torch.nn

dataset = load_dataset()
encoder, _p, _te = load_model()

# Now we train a single nn.Linear(64, 2) to predict (x, y) from the frozen embedding, MSE loss
probe = nn.Sequential(nn.Linear(64, 2))
probe.to(device)
p_loss_fn = nn.MSELoss()
p_optim = torch.optim.Adam(probe.parameters(), lr=5e-3)

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
