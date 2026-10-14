"""Generate labelled synthetic four-signal recordings for HMM experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


SIGNAL_NAMES = ("noise", "heartrate", "temperature", "motion")


@dataclass(frozen=True)
class StateProfile:
    """Gaussian emission parameters for one hidden state."""

    name: str
    means: tuple[float, float, float, float]
    standard_deviations: tuple[float, float, float, float]


RESTING_PROFILE = StateProfile(
    name="resting",
    means=(2.0, 62.0, 36.5, 0.15),
    standard_deviations=(0.35, 3.0, 0.15, 0.08),
)

ACTIVE_PROFILE = StateProfile(
    name="active",
    means=(2.3, 71.0, 36.7, 0.55),
    standard_deviations=(1.0, 8.0, 0.2, 0.45),
)

DEFAULT_PROFILES = (RESTING_PROFILE, ACTIVE_PROFILE)
DEFAULT_START_PROBABILITIES = (0.8, 0.2)
DEFAULT_TRANSITION_MATRIX = ((0.95, 0.05), (0.10, 0.90))


def generate_signal_sequence(
    length: int,
    profiles: Sequence[StateProfile] = DEFAULT_PROFILES,
    start_probabilities: Sequence[float] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
    random_state: int | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Generate observations and their hidden state labels.

    Each observation has the order ``noise, heartrate, temperature, motion``
    and can be passed directly to ``CombinedHMM.fit``.  State labels are
    returned solely for evaluating a fitted model; they are not used to train
    the unsupervised HMM.
    """
    generator = random.Random(random_state)
    return _generate_signal_sequence(
        length,
        profiles,
        start_probabilities,
        transition_matrix,
        generator,
    )


def generate_signal_dataset(
    sequence_count: int,
    sequence_length: int,
    profiles: Sequence[StateProfile] = DEFAULT_PROFILES,
    start_probabilities: Sequence[float] | None = None,
    transition_matrix: Sequence[Sequence[float]] | None = None,
    random_state: int | None = None,
) -> tuple[list[list[list[float]]], list[list[int]]]:
    """Generate independent recordings and their corresponding state labels."""
    if sequence_count < 1:
        raise ValueError("sequence_count must be at least 1")

    generator = random.Random(random_state)
    sequences: list[list[list[float]]] = []
    state_sequences: list[list[int]] = []
    for _ in range(sequence_count):
        observations, states = _generate_signal_sequence(
            sequence_length,
            profiles,
            start_probabilities,
            transition_matrix,
            generator,
        )
        sequences.append(observations)
        state_sequences.append(states)
    return sequences, state_sequences


def _generate_signal_sequence(
    length: int,
    profiles: Sequence[StateProfile],
    start_probabilities: Sequence[float] | None,
    transition_matrix: Sequence[Sequence[float]] | None,
    generator: random.Random,
) -> tuple[list[list[float]], list[int]]:
    if length < 1:
        raise ValueError("length must be at least 1")
    _validate_profiles(profiles)
    state_count = len(profiles)
    starts = _validate_probabilities(
        start_probabilities
        if start_probabilities is not None
        else [1.0 / state_count] * state_count,
        "start_probabilities",
    )
    if len(starts) != state_count:
        raise ValueError("start_probabilities must contain one value per profile")

    transitions = (
        transition_matrix
        if transition_matrix is not None
        else [[1.0 / state_count] * state_count for _ in range(state_count)]
    )
    if len(transitions) != state_count:
        raise ValueError("transition_matrix must contain one row per profile")
    validated_transitions = [
        _validate_probabilities(row, "each transition_matrix row")
        for row in transitions
    ]
    if any(len(row) != state_count for row in validated_transitions):
        raise ValueError("each transition_matrix row must contain one value per profile")

    states = [_sample_index(starts, generator)]
    observations = [_sample_observation(profiles[states[0]], generator)]
    for _ in range(1, length):
        states.append(_sample_index(validated_transitions[states[-1]], generator))
        observations.append(_sample_observation(profiles[states[-1]], generator))
    return observations, states


def _validate_profiles(profiles: Sequence[StateProfile]) -> None:
    if not profiles:
        raise ValueError("profiles must contain at least one state")
    for profile in profiles:
        if len(profile.means) != len(SIGNAL_NAMES):
            raise ValueError("each profile must provide four means")
        if len(profile.standard_deviations) != len(SIGNAL_NAMES):
            raise ValueError("each profile must provide four standard deviations")
        if not all(math.isfinite(value) for value in profile.means):
            raise ValueError("profile means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in profile.standard_deviations):
            raise ValueError("profile standard deviations must be finite and positive")


def _validate_probabilities(
    probabilities: Sequence[float],
    name: str,
) -> list[float]:
    values = [float(value) for value in probabilities]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite, non-negative probabilities")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value")
    return [value / total for value in values]


def _sample_index(probabilities: Sequence[float], generator: random.Random) -> int:
    threshold = generator.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(probabilities) - 1


def _sample_observation(profile: StateProfile, generator: random.Random) -> list[float]:
    return [
        generator.gauss(mean, standard_deviation)
        for mean, standard_deviation in zip(
            profile.means,
            profile.standard_deviations,
            strict=True,
        )
    ]