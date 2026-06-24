import numpy as np
import time
import random
from app import LittleWorld
import plotext as plt
from PIL import Image
import tempfile

plt.plot_size(width=30, height=15)


def plot_img(img, label):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        Image.fromarray(img).save(f.name)
        plt.title(label)
        plt.image_plot(f.name)
        plt.show()


LOOPS = 5_000
NUM_ACTIONS = 5
EP_LEN = 100

observations = []
states = []
actions = []

world = LittleWorld()
# state = world.render()[0].squeeze().astype("uint8") * 255
# plot_img(state, "s1")
for it in range(LOOPS):
    world.reset()
    obs, agent = world.render()
    observations.append(obs)
    states.append(agent)
    for i in range(EP_LEN - 1):
        action = random.randrange(NUM_ACTIONS)
        actions.append(action)
        obs, _done, info = world.step(action)
        observations.append(obs)
        states.append(info["agent"])
    # Append another empty action at the end to keep indexes aligned
    actions.append(4)

# print(states)

epoch = time.time()
np.savez_compressed(
    f"training_data/dataset_{epoch}",
    observations=observations,
    actions=actions,
    states=states,
)
#
