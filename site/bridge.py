"""Glue between the web page (via Pyodide) and the HMM code. Pure Python, no numpy."""

import json
import math

from hmms import CombinedHMM
from synthetic_signal_generator import SIGNAL_NAMES, generate_signal_dataset

_state = {"recordings": [], "model_result": None}


def _standardize(recording):
    columns = list(zip(*recording))
    scaled = []
    for column in columns:
        mean = sum(column) / len(column)
        std = math.sqrt(sum((v - mean) ** 2 for v in column) / len(column)) or 1.0
        scaled.append([(v - mean) / std for v in column])
    return [list(row) for row in zip(*scaled)]


def _seed(seed):
    return None if seed is None else int(seed)


def generate(count, length, seed):
    recordings, _ = generate_signal_dataset(
        sequence_count=int(count),
        sequence_length=int(length),
        random_state=_seed(seed),
    )
    raw = [[list(row) for row in recording] for recording in recordings]
    _state["recordings"] = [_standardize(r) for r in recordings]
    _state["model_result"] = None
    return json.dumps(
        {
            "signal_names": SIGNAL_NAMES,
            "raw": raw,
            "scaled": _state["recordings"],
        }
    )


def train(signals_json, n_components, seed):
    signals = json.loads(signals_json)
    columns = [SIGNAL_NAMES.index(name) for name in signals]
    data = [[[row[c] for c in columns] for row in rec] for rec in _state["recordings"]]
    model = CombinedHMM(n_components=int(n_components), random_state=_seed(seed))
    model.fit(data)
    return json.dumps(
        {
            "predicted": [model.predict(seq) for seq in data],
            "scores": [model.score(seq) for seq in data],
            "transmat": model.transmat_,
        }
    )
