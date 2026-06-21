import torch
import plotext as plt

nn = torch.nn

plt.plot_size(width=80, height=18)
# plt.theme("clear")
plt.yfrequency(0)
plt.xfrequency(0)


def plot(t: torch.Tensor, label):
    plt.title(label)
    clone = torch.clone(t)
    plt.plot(clone.cpu().detach().ravel().numpy())
    plt.show()
    plt.clear_data()


# Plot and erase
def plot_2(a: torch.Tensor, b: torch.Tensor, label):
    plt.clear_terminal(19)
    plt.title(label)
    clone_a = torch.clone(a)
    clone_b = torch.clone(b)
    plt.plot(clone_a.cpu().detach().ravel().numpy())
    plt.plot(clone_b.cpu().detach().ravel().numpy())
    plt.show()
    plt.clear_data()


device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print(f"Using {device}.")

## Initial evenly spaced -2PI to 2PI 1000 points tensor
PI = 3.1415
x = torch.linspace(start=-2 * PI, end=2 * PI, steps=1000, device=device)
plot(x, "Initial")

## Target
y = torch.sin(x).unsqueeze(1)
plot(y, "Target")

## Build the model - neural net with 6 layers
model = nn.Sequential(
    # Layers
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)
model.to(device)

## Loss and optimizer
# MSE calculates (pred - target)^2
loss_fn = nn.MSELoss()
# The optimizer stores references to the model weights and modifies them in-place on step()
optim = torch.optim.Adam(model.parameters(), lr=4 * 1e-4)

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

## Training loop
for epoch in range(2001):
    # Forward pass
    pred = model(x.unsqueeze(1))
    # Calc loss
    loss = loss_fn(pred, y)
    # Clear old gradients - they accumulate
    optim.zero_grad()
    # Backward pass (goes backward in the graph and fill in new loss gradients in the model weights)
    # This basically stores in each weight a value that tells how much it needs to change to minimize loss
    loss.backward()
    # Nudge model weights down (gradient descent)
    optim.step()
    if epoch % 20 == 0:
        with torch.no_grad():
            plot_2(y, pred, "Training")
            print(f"epoch={epoch}, loss.item()={loss.item()}")
