"""Named sets of the illustrative placeholder numbers used in env.py/pomdp.py.
None are real. Belief confidence is handled by agent.belief_update's
`confidence` weight, not by different confusion matrices - see README.
"""
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    description: str
    disassy_success_prob: float
    verify_cost: float
    inspect_cost: float
    triage_payoff: dict
    condition_obs_matrix: list  # (3, 3): rows=true y, cols=GOOD/OK/BAD


DEFAULT = Config(
    name="default",
    description="Inspecting costs more than the information is worth - go straight to Triage.",
    disassy_success_prob=0.9,
    verify_cost=1.0,
    inspect_cost=1.0,
    triage_payoff={
        "Reuse":       {"Pristine": 10.0, "Serviceable": 0.0, "Degraded": 0.0},
        "Refurbished": {"Pristine": 6.0,  "Serviceable": 5.0, "Degraded": 0.0},
        "Recycle":     {"Pristine": 2.0,  "Serviceable": 2.0, "Degraded": 1.0},
    },
    condition_obs_matrix=[
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85],
    ],
)

# Payoffs that clearly separate outcomes: GOOD -> Reuse, BAD -> Refurbished.
RELIABLE_REPAIR_VS_REUSE = Config(
    name="reliable_repair_vs_reuse",
    description="Payoffs that clearly favor Refurbish when degraded, Reuse when not.",
    disassy_success_prob=0.9,
    verify_cost=0.5,
    inspect_cost=0.3,
    triage_payoff={
        "Reuse":       {"Pristine": 12.0, "Serviceable": 1.0, "Degraded": -8.0},
        "Refurbished": {"Pristine": 4.0,  "Serviceable": 9.0, "Degraded": 7.0},
        "Recycle":     {"Pristine": 0.0,  "Serviceable": 0.5, "Degraded": 2.0},
    },
    condition_obs_matrix=[
        [0.92, 0.06, 0.02],
        [0.05, 0.90, 0.05],
        [0.02, 0.06, 0.92],
    ],
)

CONFIGS = {c.name: c for c in [DEFAULT, RELIABLE_REPAIR_VS_REUSE]}
