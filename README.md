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
- `src/agent.py`: Factored Point-Based Value Iteration (PBVI) solver. Adapts
  `AngleGrinderEnv` into explicit factored models over physical state $X$ and condition $Y$;
  `solve()` alternates Monte Carlo belief expansion (with L1-distance pruning) and
  factored Bellman backups, producing partitioned 3D alpha-vectors (`Gamma`) saved/loaded
  via `save_policy`/`load_policy`.
- `src/interface.py`: An interactive REPL that loads a saved `Gamma`, translates CAD part
  names to English, and prompts a human operator for real-world observations and confidence,
  scaling confidence into the Bayesian belief update.
- `src/main.py`: Trains a policy (`agent.solve`) and evaluates it against the
  live `AngleGrinderEnv`.
- `main.py`: Entry-point wrapper that runs `src/main.py`.
- `graph.pkl`: The pre-computed physical disassembly graph (see data note
  above).
- `models/`: Saved policies.

## Factored PBVI & State Updates

The state space factors into physical configuration $x \in X$ ($|X| = 1664$) and hidden condition $y \in Y$ ($|Y| = 3$: `Pristine`, `Serviceable`, `Degraded`).

Rather than maintaining an intractable 4992-dimensional joint belief vector and 4992D alpha-vectors, the solver uses **Factored PBVI**:
- **State Updates**: Physical state $x$ is explicitly tracked by the assembly graph (`env.current_state_id`, `most_likely_x(model)`). Physical transitions occur on `Disassy` actions (`Unscrew`, `Remove`) based on graph edges and success probability `p_success`. Condition belief $b \in \Delta^2$ is updated Bayesianly upon `Inspect` actions. `Verify` confirms the physical state $x$ without altering condition belief.
- **Factored 3D Alpha-Vectors**: Beliefs $b$ and value vectors $\alpha \in \mathbb{R}^3$ are strictly 3-dimensional over condition states $Y$. The policy partitions alpha-vectors by physical state node: $\Gamma(x) = \{(\alpha, a)\}$.
- **Continuation Bellman Backups**:
  - `Disassy`: Projects continuation across physical successor states:
    $$\alpha(y) = R(x, a, y) + \gamma \left[ p_\text{succ} \alpha^*(x_\text{succ}, y) + (1 - p_\text{succ}) \alpha^*(x_\text{fail}, y) \right]$$
  - `Inspect` / `Verify`: Evaluates continuation within the current physical node $x$, cross-summing over observation matrices $Z$:
    $$\alpha(y) = R(a, y) + \gamma \sum_o Z(y, o) \alpha_o^*(x, y)$$
  - **Dynamic Action Masking**: Continuation vectors during backups explicitly enforce `available_insp_actions`, preventing invalid action chaining (e.g. `Verify` hallucinating an immediate second `Inspect` without an intervening `Disassy`).

## Config Balancing & Confidence Scaling

### Config Balancing (`src/configs.py`)
- **Preserved Physical Costs**: Disassembly action times from `graph.pkl` are preserved as real physical durations (e.g. $-24$s per screw, $-33$s to $-51$s for housing removal).
- **Inspection & Verification Costs**: Configured to reflect realistic operational trade-offs (`inspect_cost = 28.0`, `verify_cost = 6.0`). Inspecting is more costly than a single screw removal, requiring the agent to balance diagnostic value against exploratory disassembly.
- **Calibrated Triage Payoffs (`repair_vs_reuse`)**:
  - `Reuse`: Pristine $+650.0$, Serviceable $+20.0$, Degraded $-600.0$
  - `Refurbished`: Pristine $+20.0$, Serviceable $+350.0$, Degraded $-180.0$ (reflecting the net negative return of attempting to refurbish degraded units)
  - `Recycle`: Pristine $+20.0$, Serviceable $+20.0$, Degraded $+20.0$

### Confidence Scaling (`CONFIDENCE_SCALE = 0.3`)
In Bayesian belief updating, a raw human confidence entry (e.g. $0.5$) can aggressively shift belief in just 1-2 steps, causing an agent to prematurely triage before removing key components.

A centralized constant `CONFIDENCE_SCALE = 0.3` in `src/configs.py` scales confidence into a tempered likelihood exponent:
$$b'(y) \propto b(y) \cdot P(o \mid y)^{\text{confidence} \times \text{CONFIDENCE\_SCALE}}$$
This scaling is applied consistently across:
1. Monte Carlo belief expansion rollouts (`agent.expand_beliefs`)
2. PBVI Bellman backup cross-sums (`agent._observation_cross_sum`)
3. Live evaluation rollouts (`main.py`)
4. Interactive human CLI prompts (`src/interface.py`)

This calibration ensures that operator inputs guide the policy to progressively disassemble multiple fasteners (e.g. unscrewing 3-4 screws and removing subassemblies) to gain diagnostic confidence before committing to a final triage decision.

