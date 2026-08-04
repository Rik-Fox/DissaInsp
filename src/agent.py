from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO


def create_agent(env, log_dir="./ppo_disassembly_tensorboard/"):
    """
    Creates a MaskablePPO agent.

    This agent is suitable for environments with discrete action spaces
    and where some actions may be invalid at certain steps.
    """
    agent = MaskablePPO(
        MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir
    )
    return agent


def load_agent(path, env):
    """
    Loads a pre-trained agent from a file.
    """
    agent = MaskablePPO.load(path, env=env)
    return agent
