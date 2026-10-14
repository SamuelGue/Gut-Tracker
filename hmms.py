"""A multivariate Gaussian hidden Markov model for gut-tracking signals.

Each observation is one row in the order
``noise, heartrate, temperature, motion``.  The model uses a diagonal
covariance matrix for each hidden state, which keeps training stable while
still modelling all four signals jointly through the shared state sequence.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


SIGNAL_NAMES = ("noise", "heartrate", "temperature", "motion")
_EPSILON = 1e-12


def _log_sum_exp(values: Sequence[float]) -> float:
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _normalise(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= _EPSILON:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


class CombinedHMM:
    """Train and decode a four-signal multivariate Gaussian HMM.

    Parameters:
        n_components: Number of hidden states.
        n_iter: Maximum number of Baum-Welch training iterations.
        tol: Stop when the improvement in log likelihood is below this value.
        random_state: Seed used to make parameter initialisation repeatable.
        min_covar: Lower bound for each per-signal state variance.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_iter: int = 100,
        tol: float = 1e-3,
        random_state: int | None = None,
        min_covar: float = 1e-3,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if min_covar <= 0:
            raise ValueError("min_covar must be positive")

        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self._random = random.Random(random_state)
        self.startprob_: list[float] | None = None
        self.transmat_: list[list[float]] | None = None
        self.means_: list[list[float]] | None = None
        self.covars_: list[list[float]] | None = None
        self.monitor_: list[float] = []

    def fit(
        self,
        data: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
        lengths: Sequence[int] | None = None,
    ) -> "CombinedHMM":
        """Train the model on one sequence or multiple concatenated sequences.

        ``data`` is either a sequence shaped ``(samples, 4)`` or a sequence
        of such sequences.  When multiple sequences are passed, ``lengths``
        may be omitted; when concatenated data is passed, ``lengths`` marks
        the sequence boundaries.
        """
        sequences = self._prepare_sequences(data, lengths)
        observations = [row for sequence in sequences for row in sequence]
        self._initialise(observations)
        self.monitor_ = []
        previous_log_likelihood = -math.inf

        for _ in range(self.n_iter):
            statistics = self._empty_statistics()
            log_likelihood = 0.0
            for sequence in sequences:
                log_emissions = self._log_emissions(sequence)
                alpha, sequence_log_likelihood = self._forward(log_emissions)
                beta = self._backward(log_emissions)
                self._accumulate_statistics(
                    sequence,
                    log_emissions,
                    alpha,
                    beta,
                    sequence_log_likelihood,
                    statistics,
                )
                log_likelihood += sequence_log_likelihood

            self._maximise(statistics)
            self.monitor_.append(log_likelihood)
            if log_likelihood - previous_log_likelihood < self.tol:
                break
            previous_log_likelihood = log_likelihood

        return self

    train = fit

    def decode(self, data: Sequence[Sequence[float]]) -> tuple[float, list[int]]:
        """Return the log likelihood and most likely state path for ``data``."""
        self._check_fitted()
        sequence = self._prepare_sequences(data, None)[0]
        log_emissions = self._log_emissions(sequence)
        state_count = self.n_components
        start = self.startprob_
        transitions = self.transmat_
        assert start is not None and transitions is not None

        scores = [
            math.log(start[state] + _EPSILON) + log_emissions[0][state]
            for state in range(state_count)
        ]
        backpointers: list[list[int]] = []
        for emission in log_emissions[1:]:
            next_scores: list[float] = []
            previous_states: list[int] = []
            for target in range(state_count):
                candidates = [
                    scores[source] + math.log(transitions[source][target] + _EPSILON)
                    for source in range(state_count)
                ]
                best_source = max(range(state_count), key=candidates.__getitem__)
                previous_states.append(best_source)
                next_scores.append(candidates[best_source] + emission[target])
            backpointers.append(previous_states)
            scores = next_scores

        final_state = max(range(state_count), key=scores.__getitem__)
        states = [final_state]
        for previous_states in reversed(backpointers):
            final_state = previous_states[final_state]
            states.append(final_state)
        states.reverse()
        return max(scores), states

    def predict(self, data: Sequence[Sequence[float]]) -> list[int]:
        """Return the most likely hidden state for each observation."""
        return self.decode(data)[1]

    def score(self, data: Sequence[Sequence[float]]) -> float:
        """Return the log likelihood of one observation sequence."""
        self._check_fitted()
        sequence = self._prepare_sequences(data, None)[0]
        _, log_likelihood = self._forward(self._log_emissions(sequence))
        return log_likelihood

    def _prepare_sequences(
        self,
        data: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
        lengths: Sequence[int] | None,
    ) -> list[list[list[float]]]:
        if not data:
            raise ValueError("data must contain at least one observation")
        first = data[0]
        if first and isinstance(first[0], (list, tuple)):
            sequences = [[self._validate_row(row) for row in sequence] for sequence in data]  # type: ignore[arg-type]
            if lengths is not None:
                raise ValueError("lengths must not be supplied when data contains sequences")
        else:
            rows = [self._validate_row(row) for row in data]  # type: ignore[arg-type]
            if lengths is None:
                sequences = [rows]
            else:
                if any(length < 1 for length in lengths) or sum(lengths) != len(rows):
                    raise ValueError("lengths must be positive and sum to the number of rows")
                sequences = []
                offset = 0
                for length in lengths:
                    sequences.append(rows[offset : offset + length])
                    offset += length
        if any(not sequence for sequence in sequences):
            raise ValueError("each sequence must contain at least one observation")
        return sequences

    @staticmethod
    def _validate_row(row: Sequence[float]) -> list[float]:
        if len(row) != len(SIGNAL_NAMES):
            raise ValueError("each observation must contain noise, heartrate, temperature, and motion")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("observations must contain only finite numbers")
        return values

    def _initialise(self, observations: list[list[float]]) -> None:
        feature_count = len(SIGNAL_NAMES)
        global_means = [
            sum(row[feature] for row in observations) / len(observations)
            for feature in range(feature_count)
        ]
        global_variances = [
            max(
                self.min_covar,
                sum((row[feature] - global_means[feature]) ** 2 for row in observations)
                / len(observations),
            )
            for feature in range(feature_count)
        ]
        sample_indices = list(range(len(observations)))
        self._random.shuffle(sample_indices)
        self.means_ = [
            observations[sample_indices[state % len(observations)]][:]
            for state in range(self.n_components)
        ]
        self.covars_ = [global_variances[:] for _ in range(self.n_components)]
        self.startprob_ = [1.0 / self.n_components] * self.n_components
        self.transmat_ = [
            [1.0 / self.n_components] * self.n_components
            for _ in range(self.n_components)
        ]

    def _log_emissions(self, sequence: list[list[float]]) -> list[list[float]]:
        assert self.means_ is not None and self.covars_ is not None
        constant = len(SIGNAL_NAMES) * math.log(2.0 * math.pi)
        return [
            [
                -0.5
                * (
                    constant
                    + sum(
                        math.log(self.covars_[state][feature])
                        + (row[feature] - self.means_[state][feature]) ** 2
                        / self.covars_[state][feature]
                        for feature in range(len(SIGNAL_NAMES))
                    )
                )
                for state in range(self.n_components)
            ]
            for row in sequence
        ]

    def _forward(self, emissions: list[list[float]]) -> tuple[list[list[float]], float]:
        assert self.startprob_ is not None and self.transmat_ is not None
        alpha = [[
            math.log(self.startprob_[state] + _EPSILON) + emissions[0][state]
            for state in range(self.n_components)
        ]]
        for emission in emissions[1:]:
            alpha.append([
                emission[target]
                + _log_sum_exp([
                    alpha[-1][source]
                    + math.log(self.transmat_[source][target] + _EPSILON)
                    for source in range(self.n_components)
                ])
                for target in range(self.n_components)
            ])
        return alpha, _log_sum_exp(alpha[-1])

    def _backward(self, emissions: list[list[float]]) -> list[list[float]]:
        assert self.transmat_ is not None
        beta = [[0.0] * self.n_components for _ in emissions]
        for time in range(len(emissions) - 2, -1, -1):
            beta[time] = [
                _log_sum_exp([
                    math.log(self.transmat_[state][target] + _EPSILON)
                    + emissions[time + 1][target]
                    + beta[time + 1][target]
                    for target in range(self.n_components)
                ])
                for state in range(self.n_components)
            ]
        return beta

    def _empty_statistics(self) -> dict[str, list]:
        return {
            "start": [0.0] * self.n_components,
            "transition": [[0.0] * self.n_components for _ in range(self.n_components)],
            "weight": [0.0] * self.n_components,
            "sum": [[0.0] * len(SIGNAL_NAMES) for _ in range(self.n_components)],
            "squared_sum": [[0.0] * len(SIGNAL_NAMES) for _ in range(self.n_components)],
        }

    def _accumulate_statistics(
        self,
        sequence: list[list[float]],
        emissions: list[list[float]],
        alpha: list[list[float]],
        beta: list[list[float]],
        log_likelihood: float,
        statistics: dict[str, list],
    ) -> None:
        assert self.transmat_ is not None
        gamma = [
            _normalise([
                math.exp(alpha[time][state] + beta[time][state] - log_likelihood)
                for state in range(self.n_components)
            ])
            for time in range(len(sequence))
        ]
        statistics["start"] = [
            statistics["start"][state] + gamma[0][state]
            for state in range(self.n_components)
        ]
        for time, row in enumerate(sequence):
            for state, weight in enumerate(gamma[time]):
                statistics["weight"][state] += weight
                for feature, value in enumerate(row):
                    statistics["sum"][state][feature] += weight * value
                    statistics["squared_sum"][state][feature] += weight * value * value
        for time in range(len(sequence) - 1):
            xi_values = [
                [
                    math.exp(
                        alpha[time][source]
                        + math.log(self.transmat_[source][target] + _EPSILON)
                        + emissions[time + 1][target]
                        + beta[time + 1][target]
                        - log_likelihood
                    )
                    for target in range(self.n_components)
                ]
                for source in range(self.n_components)
            ]
            normaliser = sum(map(sum, xi_values))
            if normaliser > 0:
                for source in range(self.n_components):
                    for target in range(self.n_components):
                        statistics["transition"][source][target] += (
                            xi_values[source][target] / normaliser
                        )

    def _maximise(self, statistics: dict[str, list]) -> None:
        self.startprob_ = _normalise(statistics["start"])
        self.transmat_ = [
            _normalise(row) for row in statistics["transition"]
        ]
        assert self.means_ is not None and self.covars_ is not None
        for state, weight in enumerate(statistics["weight"]):
            if weight <= _EPSILON:
                continue
            for feature in range(len(SIGNAL_NAMES)):
                mean = statistics["sum"][state][feature] / weight
                variance = (
                    statistics["squared_sum"][state][feature] / weight - mean * mean
                )
                self.means_[state][feature] = mean
                self.covars_[state][feature] = max(self.min_covar, variance)

    def _check_fitted(self) -> None:
        if self.startprob_ is None or self.transmat_ is None:
            raise RuntimeError("fit the model before decoding or scoring")
