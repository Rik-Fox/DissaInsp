import pickle
import random
from collections import defaultdict
from pathlib import Path


class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ) -> None:
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.q_table = defaultdict(float)
        self.episode_count = 0

    @staticmethod
    def _state_key(observation) -> tuple[int, ...]:
        if hasattr(observation, "tolist"):
            observation = observation.tolist()
        return tuple(int(value) for value in observation)

    def _get_valid_actions(self, action_mask=None) -> list[int]:
        if action_mask is None:
            return list(range(self.action_space_size))

        if hasattr(action_mask, "tolist"):
            action_mask = action_mask.tolist()

        return [
            index for index, is_valid in enumerate(action_mask) if is_valid
        ]

    def _get_q_value(self, state, action) -> float:
        return self.q_table[(state, action)]

    def _set_q_value(self, state, action, value) -> None:
        self.q_table[(state, action)] = value

    def act(self, observation, action_mask=None, deterministic=False) -> None | int:
        state = self._state_key(observation)
        valid_actions = self._get_valid_actions(action_mask)

        if not valid_actions:
            return None

        if deterministic or self.random.random() >= self.epsilon:
            best_value = None
            best_actions = []
            for action in valid_actions:
                value = self._get_q_value(state, action)
                if best_value is None or value > best_value:
                    best_value = value
                    best_actions = [action]
                elif value == best_value:
                    best_actions.append(action)

            return self.random.choice(best_actions)

        return self.random.choice(valid_actions)

    def predict(
        self, observation, action_mask=None, deterministic=False
    ) -> tuple[None | int, None]:
        action = self.act(
            observation, action_mask=action_mask, deterministic=deterministic
        )
        return action, None

    def learn(self, env, episodes=1000, max_steps_per_episode=50) -> QLearningAgent:
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                action = self.act(observation, action_mask=info.get("action_mask"))
                if action is None:
                    break

                next_observation, reward, terminated, truncated, next_info = env.step(
                    action
                )
                next_state = self._state_key(next_observation)
                next_valid_actions = self._get_valid_actions(
                    action_mask=next_info.get("action_mask")
                )

                current_state = self._state_key(observation)
                current_q_value = self._get_q_value(current_state, action)

                if next_valid_actions:
                    next_best_value = max(
                        self._get_q_value(next_state, next_action)
                        for next_action in next_valid_actions
                    )
                else:
                    next_best_value = 0.0

                updated_q_value = current_q_value + self.learning_rate * (
                    reward + self.discount_factor * next_best_value - current_q_value
                )
                self._set_q_value(current_state, action, updated_q_value)

                observation = next_observation
                total_reward += reward

                if terminated or truncated:
                    break

            self.episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "episode_count": self.episode_count,
            "q_table": dict(self.q_table),
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path, env=None) -> QLearningAgent:
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        agent = cls(
            action_space_size=payload["action_space_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=None,
        )
        agent.episode_count = payload.get("episode_count", 0)
        agent.q_table = defaultdict(float, payload.get("q_table", {}))
        return agent


def create_agent(env, log_dir=None) -> QLearningAgent:
    return QLearningAgent(action_space_size=env.action_space.n)


def load_agent(path, env=None) -> QLearningAgent:
    return QLearningAgent.load(path, env=env)
