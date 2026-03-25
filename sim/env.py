import numpy as np

class RANEnvironment:
    def __init__(self, num_users=10):
        self.num_users = num_users

    def reset(self):
        return self._generate_state()

    def step(self, action):
        state = self._generate_state()
        reward = self._compute_reward(action, state)
        return state, reward

    def _generate_state(self):
        # Simulate realistic SNR (signal quality)
        snr = np.random.normal(15, 5, self.num_users)
        snr = np.clip(snr, 0, 30)

        # CQI derived from SNR
        cqi = np.clip(snr / 2, 0, 15)

        # Simulate bursty traffic (queue)
        queue = np.random.poisson(lam=50, size=self.num_users)

        return {
            "snr": snr.tolist(),
            "cqi": cqi.tolist(),
            "queue": queue.tolist()
        }

    def _compute_reward(self, action, state):
        snr = np.array(state["snr"])
        queue = np.array(state["queue"])

        # Normalize values
        snr = snr / 30.0
        queue = queue / 100.0

        # Reward = prioritize users with good signal + high demand
        reward = np.sum(action * (0.6 * snr + 0.4 * queue))

        return float(reward)
