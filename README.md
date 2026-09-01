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
    Place your `graph.pkl` file in the project's root directory.

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
`Triage` aren't derived from the graph at all - they're not tied to any
particular `x`.

`Verify` and `Inspect` are each usable once per disassembly cycle: taking
one masks it out until the next `Unscrew`/`Remove` attempt, which refreshes
both again - regardless of whether that attempt actually succeeds, since
`Unscrew`/`Remove` reveal no observation, so there'd be no way to know
which happened without a `Verify` anyway. This stops the agent from just
re-querying the same sensor for more confidence instead of acting on what
it already knows or disassembling further for new information. Enforced
throughout - the live `AngleGrinderEnv` (`available_insp_actions`), the PBVI
planner in `agent.py` (each belief point carries its own
`available_insp_actions`, since two points can share a belief but differ on
which actions are still available - see `Model.valid_actions`), and
`interface.py`'s CLI (`best_action` only considers alpha-vectors whose
tagged action is currently valid for the live `AngleGrinderEnv`).

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
  above).
- `models/`: Saved policies.

All of the illustrative numbers - disassembly success probability, Verify/
Inspect costs, the condition→observation confusion matrix, and the Triage
payoff table - live in `src/configs.py`, not hardcoded in `env.py`. None are
tuned to real reliability/economics data yet.

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
| `no_inspection`            | Inspecting costs more than the information is worth - go straight to Triage. Even the *optimal* inspect-then-triage strategy scores lower than committing immediately given these numbers - don't be surprised if `src/interface.py` recommends skipping inspection entirely with this one. |
| `repair_vs_reuse`          | Payoffs that clearly separate outcomes - a `GOOD` reading leads to `Reuse`, a `BAD` reading leads to `Refurbished` (repair), a simplified version of triage where recycling is strictly dominated. |

**Performance note:** `Verify`'s observation alphabet is the full set of
physical states (it deterministically reveals `x'`), so `backup()`'s
per-observation "best previous alpha" search used to be a Python loop over
~1600 columns x every alpha-vector for that one action alone - it dominated
solve time on the real graph. It's now a single vectorized sparse-matmul
per belief point/action instead of that nested loop (same result - see
`_observation_cross_sum` in `src/agent.py`), which measured ~40-50x faster
on `graph.pkl` and should scale much better as belief/alpha sets grow.
Rebuilding each action's T/Z/R from scratch for every belief point is still
unoptimized and a candidate for a future pass.

