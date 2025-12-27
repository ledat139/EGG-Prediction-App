import matplotlib.pyplot as plt
import streamlit as st

def plot_raw_eeg(eeg):
    fig, ax = plt.subplots()
    ax.plot(eeg, linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title("Raw EEG Signal")
    st.pyplot(fig)
