# Gut Tracker

`hmms.py` contains a multivariate Gaussian hidden Markov model for these
signals, in this order:

1. noise
2. heartrate
3. temperature
4. motion

Train it with one row per time step:

```python
from hmms import CombinedHMM

samples = [
    [2.1, 65.0, 36.5, 0.2],
    [2.4, 66.0, 36.5, 0.3],
    [8.0, 110.0, 37.2, 2.0],
    [8.3, 112.0, 37.2, 2.2],
]

model = CombinedHMM(n_components=2, random_state=42)
model.fit(samples)

hidden_states = model.predict(samples)
log_likelihood = model.score(samples)
```

For multiple independent recordings, pass a list of sequences:

```python
model.fit([first_recording, second_recording])
```

Alternatively, concatenate recordings and provide their lengths:

```python
model.fit(all_samples, lengths=[len(first_recording), len(second_recording)])
```

The model uses a shared hidden state for all four signals, rather than
training four independent models.

## Synthetic training data

[synthetic_signal_generator.py](./synthetic_signal_generator.py) can create
repeatable labelled recordings using the included resting and active state
profiles:

```python
from hmms import CombinedHMM
from synthetic_signal_generator import generate_signal_dataset

recordings, true_states = generate_signal_dataset(
    sequence_count=10,
    sequence_length=100,
    random_state=42,
)

model = CombinedHMM(n_components=2, random_state=42).fit(recordings)
predicted_states = model.predict(recordings[0])
```

The generated `true_states` are useful for evaluating whether state decoding
matches the simulated state transitions. They are not supplied to `fit()`.