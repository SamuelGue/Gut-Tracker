from hmms import CombinedHMM
from synthetic_signal_generator import generate_signal_dataset
import matplotlib.pyplot as plt
import numpy as np

recordings, true_states = generate_signal_dataset(
    sequence_count=10,
    sequence_length=100,
    random_state=42,
)

def standardize(recording):
    data = np.asarray(recording, dtype=float)
    std = data.std(axis=0)
    return (data - data.mean(axis=0)) / np.where(std == 0, 1, std)


recordings = [standardize(recording).tolist() for recording in recordings]

model = CombinedHMM(n_components=2, random_state=42)
model.fit(recordings)

predicted_states = model.predict(recordings[0])

log_likelihood = model.score(recordings[0])

print("Predicted states:", predicted_states)
print("Log-likelihood:", log_likelihood)

print(model.transmat_)

fig, (ax_signal, ax_states) = plt.subplots(2, 1, sharex=True)
ax_signal.plot(recordings[0], label="Signal")
ax_signal.legend()
ax_states.step(range(len(predicted_states)), predicted_states, where="mid", color="tab:orange", label="Predicted States")
ax_states.set_yticks(sorted(set(predicted_states)))
ax_states.legend()
plt.show()

