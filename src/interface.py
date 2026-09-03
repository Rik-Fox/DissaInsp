"""Interactive REPL for testing a converged PBVI policy with a human operator."""
import argparse
from pathlib import Path
import sys

import numpy as np

from .configs import CONFIDENCE_SCALE
from .env import AngleGrinderEnv, CONDITIONS, CONDITION_OBS, INSPECTION_ACTIONS
from .agent import Model, belief_update, initial_belief, load_policy

PART_TRANSLATIONS = {
    "BG_Gehauese_p_NEU": "Main Housing",
    "BG_Handgriff": "Side Handle",
    "BG_Rotor_p": "Rotor Assembly",
    "BG_Tellerrad_V1p": "Crown Gear",
    "Getriebegehaeuse_400_V1_02p": "Gearbox Housing",
    "Kegelrad_V1_01p_NEU": "Bevel Pinion",
    "Mutter_M6_780_CT": "M6 Nut",
    "Schraube_M4x16_540_Schr1": "M4x16 Screw #1",
    "Schraube_M4x16_540_Schr2": "M4x16 Screw #2",
    "Schraube_M4x16_540_Schr3": "M4x16 Screw #3",
    "Schraube_M4x16_540_Schr4": "M4x16 Screw #4",
    "Schraube_M4x40_Schr1": "M4x40 Screw #1",
    "Schraube_M4x40_Schr2": "M4x40 Screw #2",
    "Schraube_M4x40_Schr3": "M4x40 Screw #3",
    "Schraube_M4x40_Schr4": "M4x40 Screw #4",
}


def translate_part(part_name):
    """Translates CAD part name to English."""
    return PART_TRANSLATIONS.get(part_name, part_name)


def format_state(env, state_id):
    """Formats current assembly state cleanly for display."""
    removed = env.states[state_id]["parts_removed"]
    n_removed = len(removed)
    n_total = len(env.states[state_id]["parts_present"]) + n_removed
    if n_removed == 0:
        return f"Fully Assembled (0/{n_total} parts removed)"
    parts_str = ", ".join(translate_part(p) for p in removed)
    return f"{n_removed}/{n_total} parts removed: [{parts_str}]"


def best_action(gamma, env, b):
    """Argmax over alpha-vectors valid in current environment state."""
    valid = env._get_valid_actions()
    best_value, chosen = -np.inf, None
    x_idx = env.state_to_idx[env.current_state_id]
    points = gamma.get(x_idx) or [p for pts in gamma.values() for p in pts]
    for alpha, action_id in points:
        if action_id is None or action_id not in valid:
            continue
        value = float(b @ alpha)
        if value > best_value:
            best_value, chosen = value, action_id
    return chosen


def prompt_observation(model, action_id):
    """Prompts operator for observation and confidence."""
    details = model.env.actions[action_id]
    if details["action_type"] == "Disassy":
        return 0, 1.0
    labels = CONDITION_OBS if details["type"] == "Inspect" else model.env.state_list
    while True:
        raw = input(f"Observation for '{action_id}': ").strip()
        if raw in labels:
            o_idx = labels.index(raw)
            break
    while True:
        try:
            conf = float(input("Confidence (0-1): ").strip())
            if 0.0 <= conf <= 1.0:
                return o_idx, conf
        except ValueError:
            pass


def run(policy_path, graph_path):
    """Interactive policy execution loop with human in the loop."""
    env = AngleGrinderEnv(graph_path=graph_path)
    model = Model(env)
    gamma = load_policy(policy_path)

    b = initial_belief(model)
    current_x = env.current_state_id
    last_verified_state = current_x

    while True:
        print("\n" + "=" * 60)
        print(f"Assembly State:   {format_state(env, current_x)}")
        summary = {c: round(float(p), 3) for c, p in zip(CONDITIONS, b)}
        print(f"Condition Belief: {summary}")
        print("-" * 60)

        action_id = best_action(gamma, env, b)
        if action_id is None:
            print("No action recorded for this belief - stopping.\n")
            break
        print(f"\nAGENT DECISION: Execute action '{action_id}'\n")

        details = env.actions[action_id]
        if details["action_type"] == "Triage":
            print(f"Episode complete - final triage action: {action_id}\n")
            break

        if details["action_type"] == "Disassy":
            edge = env.disassy_by_state[current_x][action_id]
            current_x = edge["next_state"]
            env.current_state_id = current_x
            env.available_insp_actions = set(INSPECTION_ACTIONS)

        elif details["type"] == "Verify":
            obs_labels = ["YES", "NO"]
            expected_removed = [translate_part(p) for p in env.states[current_x]["parts_removed"]]
            expected_desc = ", ".join(expected_removed) if expected_removed else "None (Fully Assembled)"
            print(f"Expected parts removed: [{expected_desc}]")
            while True:
                prompt_text = f"Is this physical configuration correct {obs_labels}: "
                raw = input(prompt_text).strip().upper()
                if raw in obs_labels:
                    break
                print(f"Invalid observation - choose one of {obs_labels}\n")

            if raw == "NO":
                print("\nDisassembly verification failed - reverting to last confirmed state.")
                current_x = last_verified_state
                env.current_state_id = current_x
            else:
                print("\nDisassembly confirmed successful.")
                last_verified_state = current_x

            env.available_insp_actions.discard("Verify")

        elif details["type"] == "Inspect":
            obs_labels = ["GOOD", "OK", "BAD"]
            while True:
                raw = input(f"Result of '{action_id}' {obs_labels}: ").strip().upper()
                if raw in obs_labels:
                    break
                print(f"Invalid observation - choose one of {obs_labels}\n")
            o_idx = obs_labels.index(raw)

            while True:
                raw_confidence = input("How confident are you (0-1)? ").strip()
                try:
                    confidence = float(raw_confidence)
                    if 0 <= confidence <= 1:
                        break
                except ValueError:
                    pass
                print("Enter a number between 0 and 1.\n")

            # Scales human confidence via model's shared CONFIDENCE_SCALE.
            b = belief_update(model, b, action_id, o_idx, confidence)
            env.available_insp_actions.discard("Inspect")


def main(argv=None):
    """Main CLI entry point with debugger detection."""
    parser = argparse.ArgumentParser(description="Interactive PBVI policy tester")
    parser.add_argument("--policy", required=True, help="Path to a saved Gamma policy (pickle)")
    parser.add_argument("--graph", default="graph.pkl", help="Path to graph.pkl")

    # If run without CLI arguments while attached to a debugger, use defaults
    if argv is None and len(sys.argv) == 1:
        is_debugging = sys.gettrace() is not None or "debugpy" in sys.modules
        if is_debugging:
            project_root = Path(__file__).resolve().parent.parent
            argv = [
                "--policy", str(project_root / "models" / "disassembly_policy.pkl"),
                "--graph", "graph.pkl",
            ]

    args = parser.parse_args(argv)
    run(args.policy, args.graph)


if __name__ == "__main__":
    main()
