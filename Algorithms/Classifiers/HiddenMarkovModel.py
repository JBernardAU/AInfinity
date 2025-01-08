import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        """
        Initializes the Hidden Markov Model.

        Args:
            n_states (int): Number of hidden states.
            n_observations (int): Number of observation symbols.
        """
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize transition probabilities (A), emission probabilities (B), and initial probabilities (pi)
        self.A = np.random.rand(n_states, n_states)
        self.B = np.random.rand(n_states, n_observations)
        self.pi = np.random.rand(n_states)

        # Normalize probabilities
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.pi /= self.pi.sum()

    def forward(self, observations):
        """
        Forward algorithm to calculate the probability of the observation sequence.

        Args:
            observations (list[int]): Sequence of observations.

        Returns:
            float: Probability of the observation sequence.
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, observations[t]]

        # Termination
        return np.sum(alpha[-1, :])

    def backward(self, observations):
        """
        Backward algorithm to calculate the probability of the observation sequence.

        Args:
            observations (list[int]): Sequence of observations.

        Returns:
            float: Probability of the observation sequence.
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialization
        beta[-1, :] = 1

        # Induction
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t + 1]] * beta[t + 1, :])

        # Termination
        return np.sum(self.pi * self.B[:, observations[0]] * beta[0, :])

    def viterbi(self, observations):
        """
        Viterbi algorithm to find the most probable state sequence for the observations.

        Args:
            observations (list[int]): Sequence of observations.

        Returns:
            list[int]: Most probable state sequence.
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0, :] = self.pi * self.B[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t - 1] * self.A[:, j]) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(delta[t - 1] * self.A[:, j])

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1, :])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states.tolist()

# Example usage
if __name__ == "__main__":
    # Number of states and observation symbols
    n_states = 3
    n_observations = 4

    # Observation sequence (indices of symbols)
    observations = [0, 1, 2, 3, 2, 1, 0]

    # Initialize HMM
    hmm = HiddenMarkovModel(n_states, n_observations)

    # Forward algorithm
    prob_forward = hmm.forward(observations)
    print("Probability of observation sequence (forward):", prob_forward)

    # Backward algorithm
    prob_backward = hmm.backward(observations)
    print("Probability of observation sequence (backward):", prob_backward)

    # Viterbi algorithm
    state_sequence = hmm.viterbi(observations)
    print("Most probable state sequence:", state_sequence)
