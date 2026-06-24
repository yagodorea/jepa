import os
import random
import numpy as np

GRID = 10
CELL_PX = 6  # rendering scale
OBS_SIZE = GRID * CELL_PX

ACTION_TO_DELTA = np.array(
    [
        (0, -1),  # 0 up
        (0, 1),  # 1 down
        (-1, 0),  # 2 left
        (1, 0),  # 3 right
        (0, 0),  # 4 stay
    ],
    dtype=np.int64,
)
NUM_ACTIONS = len(ACTION_TO_DELTA)


class LittleWorld:
    def __init__(self, grid=GRID, cell_px=CELL_PX, rng=None):
        self.grid = grid
        self.cell_px = cell_px
        self.obs_size = grid * cell_px
        self.rng = rng or random.Random()
        self.agent = np.zeros(2, dtype=np.int64)
        self.goal = np.zeros(2, dtype=np.int64)
        self.reset()

    # State
    def _random_cell(self):
        return np.array(
            [self.rng.randrange(self.grid), self.rng.randrange(self.grid)],
            dtype=np.int64,
        )

    def reset(self):
        """Randomizes agent and goal and returns the first observation."""
        self.agent = self._random_cell()
        self.goal = self._random_cell()
        while np.array_equal(self.goal, self.agent):
            self.goal = self._random_cell()
        return self.render()[0]

    # Control
    def step(self, action):
        """Perform action. Returns (obs, done, info)."""
        delta = ACTION_TO_DELTA[action]
        self.agent = np.clip(self.agent + delta, 0, self.grid - 1)
        done = bool(np.array_equal(self.agent, self.goal))
        info = {"agent": self.agent.copy(), "goal": self.goal.copy()}
        return self.render()[0], done, info

    # -- Layer 2: observation (pure function of state, NumPy, headless) ------
    def render(self):
        """Observation is the state rendered into a grayscale image. Draws ONLY the agent.

        Convention: first spatial axis is row (y / vertical), second is col (x / horizontal).
        """
        obs = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
        x0 = int(self.agent[0]) * self.cell_px
        y0 = int(self.agent[1]) * self.cell_px
        obs[y0 : y0 + self.cell_px, x0 : x0 + self.cell_px] = 1.0
        return obs[None, :, :], self.agent  # add channel dim -> [1, H, W]


def run_headless_benchmark(steps=200_000):
    import time

    world = LittleWorld()
    world.reset()
    goals = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        action = random.randrange(NUM_ACTIONS)
        _obs, done, _info = world.step(action)
        if done:
            goals += 1
            world.reset()
    dt = time.perf_counter() - t0
    print(f"obs shape: {world.render()[0].shape}, dtype: {world.render()[0].dtype}")
    print(f"{steps:,} steps in {dt:.3f}s -> {steps / dt:,.0f} steps/s")
    print(f"goals reached under random policy: {goals}")


# Human mode using pygame
def run_human(grid=GRID):
    import pygame
    from pygame.locals import (
        QUIT,
        KEYDOWN,
        K_UP,
        K_DOWN,
        K_LEFT,
        K_RIGHT,
        K_ESCAPE,
    )

    pygame.init()
    cell = 40
    side = grid * cell
    screen = pygame.display.set_mode((side, side))
    pygame.display.set_caption("LittleWorld")
    clock = pygame.time.Clock()

    BLACK, WHITE, GREEN = (0, 0, 0), (255, 255, 255), (0, 255, 0)
    KEY_TO_ACTION = {K_UP: 0, K_DOWN: 1, K_LEFT: 2, K_RIGHT: 3}

    world = LittleWorld(grid=grid)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key in KEY_TO_ACTION:
                    _obs, done, _info = world.step(KEY_TO_ACTION[event.key])
                    if done:
                        world.reset()

        screen.fill(BLACK)
        # human view shows the goal too, so you know where to go
        gx, gy = int(world.goal[0]), int(world.goal[1])
        ax, ay = int(world.agent[0]), int(world.agent[1])
        pygame.draw.rect(screen, GREEN, (gx * cell, gy * cell, cell, cell))
        pygame.draw.rect(screen, WHITE, (ax * cell, ay * cell, cell, cell))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


"""
Run `uv run app.py` to run test benchmark
Set HUMAN_MODE=1 to play manually
"""
if __name__ == "__main__":
    if os.getenv("HUMAN_MODE") == "1":
        run_human()
    else:
        run_headless_benchmark()
