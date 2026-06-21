import os
import torch
import torchvision as tv
import plotext as plt
from PIL import Image
import tempfile

from torch.utils.data import DataLoader
from torchvision.transforms import v2

nn = torch.nn

plt.plot_size(width=30, height=15)

# Fetch MNIST train dataset and shove it into a DataLoader (streaming read)
# Images are 28x28
print("Fetching MNIST dataset...")
mnist_train = tv.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
dl = DataLoader(mnist_train, batch_size=64, shuffle=True)

# features, labels = next(iter(dl))
# print(f"features: {features.size()}, labels: {labels.size()}")


def plot_img(img, label):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        Image.fromarray(img).save(f.name)
        plt.title(label)
        plt.image_plot(f.name)
        plt.show()


def plot_weights(model):
    weight = model[0].weight  # [16, 1, 3, 3]
    clone = torch.clone(weight)
    w = clone.cpu().detach().ravel().numpy()
    # Normalize to 0-255
    w_min, w_max = w.min(), w.max()
    w_norm = (w - w_min) / (w_max - w_min + 1e-8)
    w_uint8 = (w_norm * 255).astype("uint8")
    # Arrange the 16 images in a 4x4 grid -> [12, 12]
    # (rows, h, cols, w) -> (rows, cols, h, w) => (rows*12, cols*12)
    grid = w_uint8.reshape(4, 4, 3, 3).transpose(0, 2, 1, 3).reshape(12, 12)
    plot_img(grid, "Weights")


# Print example image
# img = features[0].squeeze()
# label = labels[0]
# plot_img(img, label)

# Model
model = nn.Sequential(
    # Starting dimension -> [1, 28, 28] (1 channel, grayscale, 28x28 pixels)
    # 2D Convolution: a 3x3 kernel scans through the (padded) image applying 16 filters (weights) to produce a 16x28x28 output
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),  # keep the strong responses
    # Halve spatial size -> takes each 2x2 grid and yields the max value there, to compact without losing too much definition
    nn.MaxPool2d(2),
    # Here, dimension is [16, 14, 14]
    # ---------
    # Another 2d convolution, bumping the feature maps to 32 -> [32, 14, 14]
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # Halve again -> [32, 7, 7]
    # ---------
    nn.Flatten(),  # Flattens 32*7*7 to [1568] so we can push it through a linear layer
    nn.Linear(1568, 10),  # Yields 10 output scores, 1 per digit
)

## Loss and optimizer
# CEL is about Classification, not Regression, it takes the raw scores and labels and compare them
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=4 * 1e-4)

# plot_weights(model)

## Training loop (pass model through each batch of images and calc loss, and backprop)
print("Training...")
for features, labels in dl:
    if os.getenv("SKIP_TRAINING"):
        break
    # Forward pass
    pred = model(features)
    # Calc loss
    loss = loss_fn(pred, labels)
    # Clear old gradients - they accumulate
    optim.zero_grad()
    # Backward pass (goes backward in the graph and fill in new loss gradients in the model weights)
    # This basically stores in each weight a value that tells how much it needs to change to minimize loss
    loss.backward()
    # Nudge model weights down (gradient descent)
    optim.step()


def extract_prediction(t: torch.Tensor) -> int:
    i = 0
    max = -999
    for idx, el in enumerate(t):
        if el > max:
            i = idx
            max = el
    return i


## Accuracy check
print("Validating accuracy...")
mnist_test = tv.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
# Test 128 random samples and calculate accuracy
test_dl = DataLoader(mnist_test, batch_size=128, shuffle=True)
test_features, test_labels = next(iter(test_dl))
pred = model(test_features)
correct = 0
for i in range(128):
    feat = test_features[i]
    img = (feat.squeeze(0).numpy() * 255).astype("uint8")
    predicted = extract_prediction(pred[i])
    if str(predicted) == str(test_labels[i].item()):
        correct += 1
    # print(f"Prediction: {predicted}. Correct: {test_labels[i].item()}")
    # plot_img(img, f"Prediction: {predicted}. Correct: {test_labels[i].item()}")

percentage = (correct / 128.0) * 100.0
print(f"Correctness = {percentage:.2f}%")

# plot_weights(model)
