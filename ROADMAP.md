# JEPA Game Project — Learning & Build Roadmap

> **Goal:** Learn the JEPA (Joint-Embedding Predictive Architecture) family by building a
> 2D navigation game and training an agent that *understands and plans in latent space*
> rather than in pixels. The game grows in difficulty across levels, and each level pulls
> us toward a new JEPA concept — ending in a **Hierarchical JEPA (H-JEPA)** that can plan
> long-term across multiple floors while avoiding enemies.
>
> **Rule of this project:** *I write the code, myself, to learn.* This doc is the map and
> the checklist. Claude guides, explains, and debugs — but does not write the implementation.

---

## 0. The Big Idea (read this first, re-read it often)

A normal predictive model (autoencoder, next-frame video predictor) tries to predict the
**raw input** — every pixel of the next frame. That's wasteful and brittle: most pixels are
unpredictable noise (exact texture, lighting), and forcing the model to predict them wastes
capacity and makes it fail on anything uncertain.

**JEPA predicts in *representation* (embedding) space instead of input space.** It learns an
encoder that maps observations to abstract vectors, and a predictor that, given the embedding
of one part of the world (the *context*), predicts the embedding of another part (the
*target*). The loss is computed between **embeddings**, never between pixels.

Why this matters for a game world model:
- The agent only needs to predict the *consequences that matter* ("I'll be one tile left,
  the enemy will be closer"), not the exact pixels.
- It can ignore unpredictable detail by simply not encoding it.
- It scales to long-horizon, abstract planning — the foundation of H-JEPA.

The three pieces, memorize them:

| Piece | Symbol | Job |
|---|---|---|
| **Context encoder** | f_θ | Encodes the visible/known part → embedding |
| **Target encoder** | f_θ̄ (often an EMA copy of f_θ) | Encodes the to-be-predicted part → target embedding |
| **Predictor** | g_φ | Predicts the target embedding from the context embedding (+ conditioning like an action) |

The single hardest problem in this whole project, and the thing JEPA is really *about*:
**representation collapse** — the trivial solution where the encoder outputs the same constant
vector for everything, making the prediction loss zero but the embeddings useless. Most of the
"tricks" in JEPA exist to prevent collapse. We will obsess over detecting and preventing it.

---

## Game ↔ Architecture Ladder (the spine of the project)

| Level | Game | JEPA concept unlocked |
|---|---|---|
| **L1** | Small walled arena, agent dot, reach a goal, discrete moves | Basic joint-embedding + **action-conditioned world model** + simple planning |
| **L2** | Bigger space, walls/obstacles, longer paths | **Multi-step rollout** prediction, error accumulation, frame-stacking |
| **L3** | Multiple floors (stairs/doors), enemies that move & shoot | **Partial observability, dynamics, uncertainty**, memory |
| **L4** | Multi-floor missions requiring long-term plans | **Hierarchical JEPA (H-JEPA)**: abstract embeddings + temporal abstraction for long-horizon planning |

Each level is "solved" when the trained agent reliably reaches the goal **using the learned
latent world model to plan**, not by hard-coded rules.

---

## Phase 0 — Foundations & Setup
*You said you're new to both deep learning and SSL, so we build the floor first. Don't skip;
but you can learn these lazily — come back as later phases demand them.*

### 0.1 Environment setup
- [X] Python 3.10+ with a virtual env (`venv` or `conda`)
- [X] Install PyTorch (CPU is fine for L1; a GPU helps from L2 on)
- [X] Install: `numpy`, `matplotlib`, `pygame` *or* `pyglet` (game rendering), `tqdm`
- [X] Verify: a script that creates a tensor, moves it to the device, runs a backward pass

### 0.2 Math you'll actually use (just-in-time, not all upfront)
- [ ] **Vectors & dot products** — embeddings *are* vectors; dot product → similarity
- [ ] **Norms (L2)** — measuring distance between two embeddings
- [ ] **Cosine similarity** — direction-based similarity, common in SSL
- [X] **Matrix multiplication** — what every linear layer does
- [X] **Gradients & the chain rule** — the intuition behind backprop (you don't derive it by hand; you must *understand* what a gradient is)
- [X] **Gradient descent / SGD / Adam** — how parameters get updated
- [X] **Mean Squared Error (L2 loss) & Smooth L1** — our prediction losses live here
- [ ] **Exponential Moving Average (EMA)** — `θ̄ ← m·θ̄ + (1−m)·θ`, how the target encoder tracks the context encoder
- [ ] **Stop-gradient** — `tensor.detach()`; why the target side must not receive gradients
- [ ] **Variance & covariance matrix** — needed for VICReg-style anti-collapse
- [ ] **Expectation / sampling** — for stochastic environments (L3+)

### 0.3 Deep learning basics (build tiny throwaway scripts to feel each one)
- [X] Tensors & autograd (`requires_grad`, `.backward()`, `.grad`)
- [X] An MLP that fits a toy function (your first training loop)
- [X] A CNN (conv, stride, padding, pooling, receptive field) classifying something tiny
- [X] The anatomy of a training loop: batch → forward → loss → `zero_grad` → `backward` → `step`
- [X] Optimizers & learning-rate schedules; overfitting & why we watch a held-out set
- [ ] (Later, optional) Vision Transformers — the *real* I-JEPA uses ViTs; we'll start with CNNs

### 0.4 Conceptual reading (skim now, return deeper later)
- [ ] **Self-supervised learning landscape**: contrastive (SimCLR) vs non-contrastive (BYOL) vs masked/predictive (MAE, I-JEPA). Know where JEPA sits: *non-contrastive, predictive, in latent space.*
- [ ] **The collapse problem** — read what BYOL and VICReg say about it
- [ ] **I-JEPA** paper (Assran et al., 2023) — the canonical image JEPA
- [ ] **V-JEPA** (video) — closest to our "predict the future" use case
- [ ] **LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)** — the H-JEPA vision, world models, the actor/configurator/cost picture. This is our north star for L4.
- [ ] **VICReg** (Bardes, Ponce, LeCun, 2021) — variance/invariance/covariance, an alternative anti-collapse recipe

**Checkpoint 0:** You can explain, out loud, (a) why predicting embeddings beats predicting
pixels, (b) what collapse is, and (c) name the three JEPA pieces and what each does.

---

## Phase 1 — Build the Game (Level 1) & Collect Data
*Before any learning model, we need the world and data from it.*

### 1.1 Define the environment (gym-style interface)
- [ ] Decide observation: render to a small image (e.g. 64×64, grayscale or RGB)
- [ ] Decide actions: discrete `{up, down, left, right, stay}`
- [ ] Decide dynamics: agent moves 1 step/tile per action, walls block movement
- [ ] Implement `reset() -> obs` and `step(action) -> (obs, done, info)`
- [ ] Render function for human viewing (sanity check the physics)

### 1.2 Collect a dataset
- [ ] Run a **random policy** for many episodes
- [ ] Save transitions as tuples **(oₜ, aₜ, oₜ₊₁)** — this triple is the heart of an action-conditioned world model
- [ ] Build a `Dataset`/`DataLoader` that serves batches of these triples
- [ ] Visualize a few triples to confirm they're correct (action actually causes the change)

**Checkpoint 1:** You can sample a batch of (oₜ, aₜ, oₜ₊₁) and eyeball that the transition
makes sense.

---

## Phase 2 — First Joint Embedding (no actions yet)
*Goal: get an encoder + predictor training **without collapsing**, on a simpler task than
full dynamics. Here the "context" and "target" are two views/parts of the **same** frame
(e.g. masked patches), à la I-JEPA. This isolates the collapse problem from the dynamics
problem.*

### 2.1 Build the pieces
- [ ] **Encoder** (small CNN): image → embedding vector (or grid of patch embeddings)
- [ ] **Target encoder**: start as an EMA copy of the encoder, updated each step, **stop-grad**
- [ ] **Predictor** (small MLP/transformer): predicts target embedding from context embedding
- [ ] **Loss**: L2 / smooth-L1 between predicted and target embeddings (NOT pixels)

### 2.2 Train and — most importantly — watch for collapse
- [ ] Implement the EMA update for the target encoder
- [ ] Train; log the **per-dimension standard deviation of embeddings** across a batch
- [ ] Collapse alarm: if embedding std → ~0, you've collapsed. Understand *why* it happened.
- [ ] (Ablation to internalize the lesson) Try training **without** EMA/stop-grad and watch it collapse on purpose.

### 2.3 Prove the embeddings are meaningful
- [ ] Train a **linear probe**: freeze the encoder, fit a tiny linear layer that decodes the
      agent's (x, y) position from the embedding. If position decodes well, the latent is real.

**Checkpoint 2:** Embeddings don't collapse (std stays healthy) **and** a linear probe recovers
agent position. You now have a working encoder.

---

## Phase 3 — Action-Conditioned World Model (solve L1's "prediction")
*Now the predictor predicts the **future**: given the embedding of oₜ and the action aₜ,
predict the embedding of oₜ₊₁.*

- [ ] Encode oₜ → zₜ (context encoder), encode oₜ₊₁ → z̄ₜ₊₁ (target encoder, stop-grad)
- [ ] **Condition the predictor on the action**: ẑₜ₊₁ = g_φ(zₜ, aₜ) (embed the discrete action first)
- [ ] Loss = distance(ẑₜ₊₁, z̄ₜ₊₁) in latent space
- [ ] Keep watching collapse metrics — dynamics make collapse *easier*, stay vigilant
- [ ] Evaluate: latent prediction error, and a linear probe that decodes position from ẑₜ₊₁
- [ ] (Optional sanity) Train a small *decoder* (separate, frozen world model) just to visualize
      what the predicted latent "looks like" — purely for your intuition, not part of JEPA

**Checkpoint 3:** Given a state and an action, the model predicts the next state's embedding
accurately (low latent error, position decodes correctly from the prediction).

---

## Phase 4 — Plan in Latent Space → actually beat Level 1
*A world model is only useful if it drives behavior. We plan entirely in the learned latent
space.*

- [ ] Represent the **goal** as an embedding (encode the goal observation)
- [ ] Define a **cost**: distance in latent space between predicted future state and goal
- [ ] Implement a planner — start simple:
  - [ ] **Random shooting**: sample many action sequences, roll them out *in latent space*
        using the predictor, pick the sequence with lowest cost, execute the first action (MPC)
  - [ ] (Upgrade) **CEM / MPPI** for smarter sampling
- [ ] Close the loop: at each step, plan → act → observe → re-plan
- [ ] Measure success rate of reaching the goal

**Checkpoint 4 — LEVEL 1 SOLVED:** The agent reliably reaches the goal by planning with the
learned latent world model. You've built a complete minimal JEPA agent.

---

## Phase 5 — Level 2: Bigger World, Longer Horizons
*Bigger arenas and obstacles mean plans are longer, so single-step prediction isn't enough —
errors compound. This is where multi-step rollout discipline matters.*

- [ ] Extend the env: larger space, walls/obstacles, longer optimal paths
- [ ] Train the predictor to roll out **multiple steps** in latent space
  - [ ] Teacher forcing first (feed real next embeddings), then **free-running** (feed your own
        predictions) — observe error accumulation
- [ ] Study & log **how prediction error grows with horizon**
- [ ] Add **frame-stacking** (or two consecutive frames) so the encoder can perceive velocity /
      recent motion — your first taste of partial observability
- [ ] Re-run planning with longer horizons; retune planner

**Checkpoint 5 — LEVEL 2 SOLVED:** Agent navigates the larger obstacle world; you can plot and
explain the prediction-error-vs-horizon curve.

---

## Phase 6 — Level 3: Floors, Enemies, Uncertainty
*Enemies that move and shoot make the future **stochastic and partially observable**. This is
where JEPA's "predict representations, not pixels" really earns its keep: it can predict the
*distribution of outcomes that matters* without hallucinating exact pixels.*

- [ ] Extend the env: multiple floors connected by stairs/doors; moving enemies; projectiles; death/respawn
- [ ] Handle **partial observability**: add memory — frame stacks, or an RNN/GRU, or a small
      transformer over recent embeddings
- [ ] Confront **uncertainty / multimodality**: the future isn't deterministic now. Study how
      JEPA copes (it predicts an embedding that captures the predictable part; discuss
      latent-variable / informational approaches, and why pixel predictors blur here)
- [ ] Keep collapse metrics + position/enemy probes running; introduce a probe for "is there a
      threat nearby" to confirm the latent encodes danger
- [ ] Planning now balances reaching the goal **and** avoiding enemies (cost includes threat)

**Checkpoint 6 — LEVEL 3 SOLVED:** Agent survives and reaches goals on a single floor with
enemies; latent provably encodes threat info.

---

## Phase 7 — Level 4: Hierarchical JEPA (H-JEPA) for Long-Term Planning
*The capstone, and the reason we built this game. Long multi-floor missions are too long to
plan step-by-step in the low-level latent. We stack JEPAs: a **high-level** module that
produces abstract embeddings over **coarse time scales** (e.g. "get to the stairs", "clear the
room"), and the **low-level** world model from earlier that executes short-horizon control.*

- [ ] Study LeCun's H-JEPA section again now that you have the low level working
- [ ] Build a **high-level encoder/predictor** operating on **temporally abstract** states
      (e.g. encode segments of trajectory; predict over many steps / subgoal transitions)
- [ ] Define **subgoals** as high-level latent targets the low level tries to reach
- [ ] **Hierarchical planning**: high level plans a sequence of subgoal embeddings over the long
      horizon; low level plans actions to reach each subgoal
- [ ] Make sure higher levels are *more abstract* (coarser time, less detail) — verify with probes
- [ ] Evaluate on missions that are **impossible to solve** with the flat (single-level) planner
      from Phase 4–6 due to horizon length

**Checkpoint 7 — PROJECT GOAL:** A two-level (or more) JEPA agent completes long multi-floor
missions by planning abstractly up top and controlling precisely down below.

---

## Phase 8 — Analysis, Ablations & Write-Up
*Where the real understanding crystallizes.*

- [ ] **Ablations** (each teaches a lesson):
  - [ ] Remove EMA/stop-grad → watch collapse
  - [ ] Swap EMA approach for **VICReg** (variance + covariance regularization) → compare
  - [ ] Replace latent loss with **pixel reconstruction** baseline → compare robustness & speed
  - [ ] Remove hierarchy at L4 → show flat planner fails on long horizons
- [ ] **Visualize the latent space**: t-SNE / UMAP of embeddings, colored by position/floor/threat
- [ ] **Probing suite**: what info is (and isn't) linearly decodable at each level of the hierarchy
- [ ] Write a short report: what worked, what collapsed, what surprised you

---

## Concepts Glossary (fill in as you learn — in your own words)
- **JEPA** —
- **Context / target / predictor** —
- **Representation collapse** —
- **EMA target encoder** —
- **Stop-gradient** —
- **Latent (embedding) loss** —
- **Action-conditioned prediction** —
- **MPC / random shooting / CEM / MPPI** —
- **VICReg (variance/invariance/covariance)** —
- **Linear probe** —
- **Temporal abstraction / subgoal** —
- **H-JEPA** —

## Key Resources
- I-JEPA — Assran et al., 2023 ("Self-Supervised Learning from Images with a JEPA")
- V-JEPA — Meta AI, 2024 (video JEPA)
- LeCun — "A Path Towards Autonomous Machine Intelligence", 2022 (H-JEPA, world models)
- VICReg — Bardes, Ponce, LeCun, 2021
- BYOL — Grill et al., 2020 (EMA + predictor anti-collapse, the trick I-JEPA inherits)
- (Control/world-model context) Dreamer / latent world models — for comparison of latent
  planning ideas (note: those reconstruct pixels; we deliberately don't)

---

## Progress Log
*Date — what I did — what I learned — what's blocking me.*

- 2026-06-15 — Created roadmap; chose 2D multi-level navigation game with H-JEPA endgame. —
  Understood the game↔architecture ladder. — Next: Phase 0 setup.
