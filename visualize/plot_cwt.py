import matplotlib.pyplot as plt
import streamlit as st
from preprocess.cwt import compute_cwt

def plot_cwt(eeg):
    coef = compute_cwt(eeg)

    fig, ax = plt.subplots()
    ax.imshow(
        coef,
        aspect="auto",
        cmap="jet",
        origin="lower"
    )
    ax.set_title("CWT Scalogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")

    st.pyplot(fig)
