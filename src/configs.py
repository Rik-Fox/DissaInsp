"""Named sets of the illustrative placeholder numbers used in env.py/pomdp.py.
None are real. Belief confidence is handled by agent.belief_update's
`confidence` weight, not by different confusion matrices - see README.
"""
from dataclasses import dataclass

# Tunable multiplier mapping confidence (0-1) to Bayesian evidence exponent.
CONFIDENCE_SCALE = 0.3


@dataclass
class Config:
    name: str
    description: str
    disassy_success_prob: float
    verify_cost: float
    inspect_cost: float
    triage_payoff: dict
    condition_obs_matrix: list  # (3, 3): rows=true y, cols=GOOD/OK/BAD


NO_INSPEC = Config(
    name="no_inspection",
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

# Payoffs that clearly separate outcomes: GOOD -> Reuse, BAD -> Refurbished, Recycle is strictly dominated
REPAIR_VS_REUSE = Config(
    name="repair_vs_reuse",
    description="Payoffs that balance real disassembly costs against diagnostic confidence.",
    disassy_success_prob=0.9,
    verify_cost=6,
    inspect_cost=28,
    triage_payoff={
        "Reuse":       {"Pristine": 650.0, "Serviceable": 20.0,  "Degraded": -600.0},
        "Refurbished": {"Pristine": 20.0,  "Serviceable": 350.0, "Degraded": -180.0},
        "Recycle":     {"Pristine": 20.0,  "Serviceable": 20.0,  "Degraded": 20.0},
    },
    condition_obs_matrix=[
        [0.75, 0.15, 0.1],
        [0.125, 0.75, 0.125],
        [0.1, 0.15, 0.75],
    ],
)

CONFIGS = {c.name: c for c in [NO_INSPEC, REPAIR_VS_REUSE]}

# Alias: tests/back-compat references expect a `DEFAULT` config.
DEFAULT = NO_INSPEC
