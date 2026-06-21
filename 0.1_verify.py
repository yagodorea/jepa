import torch

# Get device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}.")

x = torch.tensor(data=[1.0, 2.0, 3.0], requires_grad=True, device=device)
print(f"x={x}")

# Forward pass
y = (x**2).sum()
print(f"y={y} -> Apply (x**2).sum() to each element")

# Backward pass
print("->> y.backward")
y.backward()

print(
    f"x.grad={x.grad} -> Run backwards, filling each element's gradient f(x) = 2x (derivative fn)"
)
