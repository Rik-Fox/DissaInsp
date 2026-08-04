# Disassembly Sequence Learning with Reinforcement Learning

This project implements a Reinforcement Learning (RL) agent using tabular Q-learning to learn optimal disassembly sequences for a product. The environment is modeled as a state graph where states represent the assembly configuration and actions are disassembly operations.

## Project Structure

- `src/agent.py`: Defines the tabular Q-learning agent and its save/load helpers.
- `src/env.py`: Implements the custom Gymnasium environment `AngleGrinderEnv`, which interacts with a pre-computed state graph. It handles state transitions, rewards, and action masking.
- `src/main.py`: Contains the main training and evaluation logic for the RL agent.
- `main.py`: Small entry-point wrapper that runs the package-based implementation from `src/`.
- `graph.pkl` or `ui.html`: (External data) The state graph data for the disassembly problem. `graph.pkl` is preferred, but `ui.html` can be parsed as a fallback.
- `disassembly_graph/`: Contains the bundled disassembly example assets and archives.
- `models/`: Directory to save trained RL models.
- `ppo_disassembly_tensorboard/`: Directory for TensorBoard logs during training.

## Features

- **Custom Gymnasium Environment**: `AngleGrinderEnv` models the disassembly process, providing observations (removed parts), rewards (based on time and goal/dead-end states), and action masks.
- **Tabular Q-learning Agent**: Learns a state-action value table and masks invalid actions so the agent only considers legal disassembly steps.
- **Training and Evaluation**: Scripts to train a new agent from scratch and evaluate its performance.
- **Persisted Q-table**: Saves the learned Q-table to disk so it can be loaded later for evaluation.

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
    pip install stable-baselines3 sb3-contrib gymnasium numpy pickle-mixin
    ```

4.  **Prepare Graph Data:**
    Place your `graph.pkl` file in the project's root directory. If `graph.pkl` is not available, the system will attempt to parse the bundled `ui.html` from `disassembly_graph/disassembly_angle_grinder/disassembly_angle_grinder/ui.html` if it's present.

## Usage

To train a new agent and then evaluate it, simply run the `main.py` script:

```bash
python main.py
```

This will:
1.  Train a tabular Q-learning agent for `2000` episodes.
2.  Save the learned Q-table to `./models/disassembly_agent_q_table.pkl`.
3.  Evaluate the trained agent by running one episode and printing the actions taken and rewards received.

You can adjust the episode count or step limit in `src/main.py` for longer or shorter training runs.

## TensorBoard

During training, TensorBoard logs are generated in the `ppo_disassembly_tensorboard/` directory. You can view them by running:

```bash
tensorboard --logdir ppo_disassembly_tensorboard/
```
Then open your web browser to the address provided by TensorBoard (usually `http://localhost:6006`).