import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from hmms import CombinedHMM
from synthetic_signal_generator import SIGNAL_NAMES, generate_signal_dataset

st.set_page_config(page_title="Your Gut Tracker", layout="wide")

# Palette and fonts taken from suna.health
CORAL = "#FE8370"
DARK_CORAL = "#D9573F"
BONE = "#F3ECDD"
BONE_LIGHT = "#FAF4E4"
NAVY = "#2A3744"
SLATE = "#4E6876"
BORDER = "#D9CFB8"
LINE_COLOURS = [CORAL, NAVY, "#7D98A6", DARK_CORAL]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Host+Grotesk:wght@400;500;600;700&family=Averia+Serif+Libre:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"], .stMarkdown p, label {{ font-family: 'Host Grotesk', sans-serif; color: {NAVY}; }}
    h1 {{ font-family: 'Averia Serif Libre', serif !important; font-weight: 700 !important; color: {NAVY}; font-size: 3.2rem !important; letter-spacing: -0.02em; }}
    h2 {{ font-family: 'Averia Serif Libre', serif !important; font-weight: 400 !important; color: {NAVY}; font-size: 1.6rem !important; }}
    h3 {{ font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; color: {DARK_CORAL}; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.12em; }}
    [data-testid="stCaptionContainer"], .stSelectbox label, .stNumberInput label, .stMultiSelect label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: {SLATE}; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; background: {BONE_LIGHT}; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border: 1px solid {BORDER} !important; border-radius: 20px; background: {BONE_LIGHT}; padding: 1rem 1.5rem; }}
    .stButton > button {{ border-radius: 999px; border: 1px solid {NAVY}; background: transparent; color: {NAVY}; font-family: 'Host Grotesk', sans-serif; font-weight: 600; }}
    .stButton > button[kind="primary"] {{ background: {CORAL}; border-color: {CORAL}; color: {NAVY}; }}
    .stButton > button:hover {{ background: {NAVY}; color: {BONE}; border-color: {NAVY}; }}
    [data-testid="stPyplot"] img, [data-testid="stImage"] img {{ border-radius: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.facecolor": BONE_LIGHT,
        "axes.facecolor": BONE_LIGHT,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "font.family": "sans-serif",
        "axes.prop_cycle": plt.cycler(color=LINE_COLOURS),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": BORDER,
    }
)

st.title("Your Gut Tracker")

with st.container(border=True):
    st.write("Welcome to Your Gut Tracker! This is your one-stop-shop for tracking and analysing your gut health.")
    st.header("\'Your wearables track your heart, your sleep, your steps. But nothing tracks your gut.\'")
    st.write("""This is what sent me down the rabbit hole of what it even means to track your gut, and it shows that so few people actually 
know what to look for. Suna has been on my radar for a while, and it has led me to explore what it takes to understand what happens in the gut.""")
with st.container(border=True):
    st.header("How do we know when something is out of balance?")
    st.write("""Daily tracking is an important part of understanding your gut health. However, what do you do when something seems off? Maybe you changed your 
diet and now your daily score is changing. How do you interpret these changes? It's not an easy question to answer, your gut certainly won't be the same 
for your whole life, and a change in lifestyle shouldn't always be a cause for concern.""")
with st.container(border=True):
    st.header("The Model")
    st.write("""Everyone's gut is different, and so having a unique profile for your gut is essential. However, one profile can only capture so much. 
Your \'normal\' might change from day to day, week to week, or even season to season, and your gut health tracker should adapt accordingly. The 
purpose of this model is to introduce the technique of Hidden Markov Models (HMMs) to gain an insight into how your normal changes over time.""")
with st.container(border=True):
    st.header("The Signals")
    st.write("""The gut communicates with the rest of the body, often without anyone noticing. Your sleep schedule, your diet, your physical activity
all have impacts. Suna measures noise, heartrate, temperature, motion, and many more things to build a picture bespoke to you.""")
with st.container(border=True):
    st.header("How does it work?")
    st.write("""A Hidden Markov Model is a statistical way to pull out hidden patterns from sequential data. It allows us to learn what kind of state our 
gut is in over long periods of time. It differs from a traditional model in that it considers all the past signals when deciding the current state. 
You can say it learns your gut's normal patterns over time.""")

def standardize(recording):
    data = np.asarray(recording, dtype=float)
    std = data.std(axis=0)
    return (data - data.mean(axis=0)) / np.where(std == 0, 1, std)


def transmat_latex(matrix):
    rows = r" \\ ".join(" & ".join(f"{value:.3f}" for value in row) for row in matrix)
    return r"P = \begin{bmatrix}" + rows + r"\end{bmatrix}"


with st.sidebar:
    st.header("Data")
    signals = st.multiselect("Signals", SIGNAL_NAMES, default=list(SIGNAL_NAMES))
    sequence_count = st.number_input("Recordings", 1, 100, 10)
    sequence_length = st.number_input("Samples per recording", 2, 5000, 100)
    seed = st.number_input("Random seed", 0, 1_000_000, 42)
    if st.button("Generate new data", use_container_width=True):
        recordings, true_states = generate_signal_dataset(
            sequence_count=int(sequence_count),
            sequence_length=int(sequence_length),
            random_state=int(seed),
        )
        st.session_state.recordings = recordings
        st.session_state.true_states = true_states
        st.session_state.pop("model_result", None)

    st.header("Model")
    n_components = st.number_input("Hidden states", 2, 10, 2)
    if st.button("Train new model", type="primary", use_container_width=True):
        if "recordings" not in st.session_state:
            st.warning("Generate data first.")
        elif not signals:
            st.warning("Select at least one signal.")
        else:
            columns = [SIGNAL_NAMES.index(name) for name in signals]
            scaled = [
                standardize(recording)[:, columns].tolist()
                for recording in st.session_state.recordings
            ]
            model = CombinedHMM(n_components=int(n_components), random_state=int(seed))
            model.fit(scaled)
            st.session_state.model_result = {
                "signals": list(signals),
                "scaled": scaled,
                "predicted": [model.predict(sequence) for sequence in scaled],
                "scores": [model.score(sequence) for sequence in scaled],
                "transmat": model.transmat_,
            }

if "recordings" not in st.session_state:
    st.info("Use the sidebar to generate data.")
    st.stop()

result = st.session_state.get("model_result")
shown_signals = result["signals"] if result else signals
columns = [SIGNAL_NAMES.index(name) for name in shown_signals]
plot_box = st.container(border=True)
matrix_box = st.container(border=True)
with plot_box:
    index = st.selectbox(
        "Recording",
        range(len(st.session_state.recordings)),
        format_func=lambda i: f"Recording {i + 1}",
    )

recording = np.asarray(st.session_state.recordings[index], dtype=float)
rows = 2 if result else 1
if not columns:
    st.warning("Select at least one signal.")
    st.stop()

fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(10, 3 * rows), squeeze=False)
axes = axes[:, 0]
scaled_recording = standardize(recording)
for column, name in zip(columns, shown_signals):
    axes[0].plot(scaled_recording[:, column], label=name)
axes[0].set_ylabel("Standardized value")
axes[0].legend(loc="upper right", facecolor=BONE_LIGHT, edgecolor=BORDER)
if result:
    predicted = result["predicted"][index]
    axes[-1].step(range(len(predicted)), predicted, where="mid", color=CORAL, linewidth=2)
    axes[-1].set_yticks(sorted(set(predicted)))
    axes[-1].set_ylabel("Predicted state")
axes[-1].set_xlabel("Sample")
fig.tight_layout()
with plot_box:
    st.pyplot(fig)
plt.close(fig)

if result:
    with matrix_box:
        st.subheader("Transition matrix")
        st.latex(transmat_latex(result["transmat"]))
        st.caption(f"Log-likelihood of this recording: {result['scores'][index]:.2f}")
