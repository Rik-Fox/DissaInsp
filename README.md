# Disassembly & Triage POMDP

This project models a robotic disassembly + condition-triage task as a
**Partially Observable Markov Decision Process (POMDP)**, and provides a
Gymnasium environment for simulating/training against it.

> **⚠️ Data note:** in the bundled `graph.pkl`, the part
> `Mutter_M6_780_CT` (a nut) never appears as the `target_part` of any edge -
> there is no action anywhere in the graph that removes it. This looks like
> a gap in how the graph was generated rather than something intentional.
> Every other part is reachable.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd DissaInsp
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Graph Data:**
    Place your `graph.pkl` file in the project's root directory. If `graph.pkl` is not available, the system will attempt to parse the bundled `ui.html` from `disassembly_graph/disassembly_angle_grinder/disassembly_angle_grinder/ui.html` if it's present.

## Running

Train a policy and evaluate it against the live environment:
```bash
python main.py
```

Test a saved policy interactively, supplying observations yourself in place
of a real operator:
```bash
python -m src.interface --policy models/disassembly_policy.pkl --graph graph.pkl
```

## The model

The state is a pair `(x, y)`:
- **x - physical configuration**: which parts have been removed so far.
  Backed by a pre-computed graph (`graph.pkl`) where each node is the sorted
  set of parts still present, and each edge is a single physical disassembly
  step (e.g. unscrewing one particular screw).
- **y - hidden condition**: `Pristine` / `Serviceable` / `Degraded`. Not
  represented in the graph at all - it's sampled per episode and never
  directly observed, only inferred through inspection.

Actions are grouped into three categories:
| Category  | Actions                          | Effect                                                      |
|-----------|-----------------------------------|--------------------------------------------------------------|
| `Disassy` | `Unscrew`, `Remove`               | changes `x`, reveals nothing (null observation)               |
| `Insp`    | `Verify`, `Inspect`               | leaves `x` unchanged; `Verify` reveals `x`, `Inspect` reveals a noisy signal about `y` |
| `Triage`  | `Reuse`, `Refurbished`, `Recycle` | ends the episode with a payoff that depends on `y`             |

That's 7 actions total. `Unscrew`/`Remove` aren't one-action-per-part:
since a graph node's identity is just the *set* of parts still present, the
order parts are removed in doesn't matter, so each is a single action that
resolves deterministically (a fixed tie-break rule, e.g. "unscrew whichever
remaining screw sorts first") given the current `x`. `Verify`/`Inspect`/
`Triage` aren't derived from the graph at all - they're always available,
regardless of `x`.

Only `Triage` ever produces a positive reward; every other action just costs
its own time. Once disassembly is exhausted, `Unscrew`/`Remove` simply become invalid (masked
out) and the agent is expected to inspect/verify and then triage.

## Project structure

- `src/pomdp.py`: The formal POMDP model - sparse transition (`T`),
  observation (`Z`), and reward (`R`) builders (`TransitionModel`,
  `ObservationModel`, `RewardModel`). This is the shared source of truth for
  T/Z/R, used by both the simulator and the PBVI solver below.
- `src/configs.py`: Named `Config`s bundling every illustrative number
  (success probability, costs, sensor reliability, Triage payoffs).
- `src/env.py`: `AngleGrinderEnv`, a Gymnasium environment that simulates the
  model above - the input/output interface an agent actually interacts
  with. Loads `graph.pkl` + a `Config`, builds the 7-action space, and calls
  into `pomdp.py` for every reward/transition/observation calculation so the
  simulator and the planner agree on semantics.
- `src/agent.py`: A Point-Based Value Iteration (PBVI) solver - `Model` adapts
  an `AngleGrinderEnv`'s per-action T/Z/R into the matrices a POMDP solver
  needs; `solve()` alternates Monte Carlo belief expansion (with L1-distance
  pruning) and point-based value backups, producing a set of alpha-vectors
  (`Gamma`) that can be saved/loaded via `save_policy`/`load_policy`. A draft
  reference implementation, not a tuned/optimized solver - see the
  performance note below.
- `src/interface.py`: An interactive REPL that loads a saved `Gamma`, picks
  the best action for the live belief, and - for Verify/Inspect - prompts a
  human operator for the real-world observation and their confidence in it,
  feeding both back through the same belief update used during training.
- `src/main.py`: Trains a policy (`agent.solve`) and evaluates it against the
  live `AngleGrinderEnv`.
- `main.py`: Entry-point wrapper that runs `src/main.py`.
- `graph.pkl`: The pre-computed physical disassembly graph (see data note
  above). `disassembly_graph/` holds the bundled example assets it can also
  be parsed from as a fallback.
- `models/`, `ppo_disassembly_tensorboard/`: Saved models / training logs.

All of the illustrative numbers - disassembly success probability, Verify/
Inspect costs, the condition→observation confusion matrix, and the Triage
payoff table - live in `src/configs.py`, not hardcoded in `env.py`. None are
tuned to real reliability/economics data.

Every config uses the *same* mechanism for Inspect: one fixed confusion
matrix, Bayes-updated from a uniform `[1/3, 1/3, 1/3]` prior. How much a
single reading moves that belief isn't a property of the config, though - it's
a `confidence` weight (0-1) passed into `agent.belief_update` at update time
(a "tempered" Bayes update, `b'(y) ∝ b(y) · P(o|y)^confidence`). `confidence=1`
is a full update, `confidence=0` ignores the reading entirely. `src/interface.py`
asks a human operator for this alongside their observation; during training,
`agent.expand_beliefs` samples it randomly per reading, so the learned policy
sees belief points reflecting a whole range of possible operator confidence,
not just full trust.

| Config                     | What it shows                                                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| `default`                  | Inspecting costs more than the information is worth - go straight to Triage. Even the *optimal* inspect-then-triage strategy scores lower than committing immediately given these numbers - don't be surprised if `src/interface.py` recommends skipping inspection entirely with this one. |
| `reliable_repair_vs_reuse` | Payoffs that clearly separate outcomes - a `GOOD` reading leads to `Reuse`, a `BAD` reading leads to `Refurbished` (repair), mirroring a realistic triage call. |

**Performance note:** `src/agent.py`'s backup is unoptimized - it's
noticeably slower for `Verify`, since its observation alphabet is the full
set of physical states (`Verify` deterministically reveals `x'`), so the
backup loops over ~1600 columns for that one action on the real graph. Fine
for a draft/small runs, but expect it to dominate solve time at scale.

## Usage

```python
from src.env import AngleGrinderEnv

env = AngleGrinderEnv(graph_path="graph.pkl", config="reliable_repair_vs_reuse")
observation, info = env.reset()  # samples a hidden ground-truth condition y
action = info["action_mask"].argmax()  # any valid action, e.g. via the mask
observation, reward, terminated, truncated, info = env.step(action)
```

To solve a policy offline and test it interactively:

```python
from src.env import AngleGrinderEnv
from src.agent import Model, solve, save_policy

env = AngleGrinderEnv(graph_path="graph.pkl")
gamma = solve(Model(env), n_iterations=5, n_trajectories=20, horizon=10)
save_policy(gamma, "policy.pkl")
```

## Tests

```bash
python -m unittest discover -s tests
```

## TensorBoard

During training, TensorBoard logs are generated in the `ppo_disassembly_tensorboard/` directory. You can view them by running:

```bash
tensorboard --logdir ppo_disassembly_tensorboard/
```
Then open your web browser to the address provided by TensorBoard (usually `http://localhost:6006`).
