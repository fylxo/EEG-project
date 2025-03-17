import os
import json
import mne
import scipy

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert, coherence


import my_functions as my_fun




# ------------------------------------------------------------------------------------- RAW DATA VISUALIZATION -------------------------------------------------------------------------------------

def plot_raw(raw, title="EEG Signal"):
    # Input: mne.io.Raw Object
    data, times = raw[:, :]
    plt.figure(figsize=(12, 6))
    for i in range(min(10, data.shape[0])):  # first 10 channels
        plt.plot(times, data[i] * 1e6 + i * 100, label=raw.ch_names[i])  # µV scale
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (µV)")
    plt.legend(loc="upper right")
    plt.show()



def plot_raw_eeg(epochs, subject, condition):
    # Input: mne.Epochs Object
    raw = epochs.get_data()[:, :10, :] * 1e6  # Convert to µV for plotting only
    times = epochs.times
    plt.figure(figsize=(12, 6))
    
    for i in range(raw.shape[1]):
        plt.plot(times, raw[0, i, :] + i * 20, label=f"Ch {epochs.ch_names[i]}")  # Offset for readability

    plt.xlabel("Time (s)")
    plt.ylabel("EEG Amplitude (µV)")
    plt.title(f"Raw EEG Signal - {subject} ({condition})")
    plt.legend()
    plt.show()



def plot_erd_ers(epochs, subject, condition, fmin=6, fmax=30):
    """
    Plot ERD/ERS for motor-related channels, accounting for variations in channel names.
    """
    erd_ers, times, freqs = my_fun.compute_erd_ers(epochs, fmin=fmin, fmax=fmax)
    available_channels = epochs.ch_names  # Get available EEG channels

    plt.figure(figsize=(10, 6))

    # Define motor cortex channels (handle extra characters in names)
    motor_channels = ["C6", "CP5"]
    matched_channels = motor_channels
    #matched_channels = [ch for ch in available_channels if any(target in ch for target in motor_channels)]

    if not matched_channels:
        print("⚠️ No valid motor channels found in dataset!")
        return

    for ch in matched_channels:
        ch_idx = available_channels.index(ch)

        # ✅ FIX: Compute mean over frequency axis while keeping the correct time shape
        freq_mask = np.logical_and(freqs >= fmin, freqs <= fmax)  # Ensure boolean mask
        erd_curve = np.mean(erd_ers[:, ch_idx, freq_mask, :], axis=(0, 1))  # Mean over epochs & frequency

        if erd_curve.shape[0] != times.shape[0]:  # Debugging check
            print(f"⚠️ Dimension mismatch: times.shape={times.shape}, erd_curve.shape={erd_curve.shape}")

        plt.plot(times, erd_curve, label=f"{ch}")

    plt.axvline(0, color='k', linestyle='--', label="Movement Onset")
    plt.xlabel("Time (s)")
    plt.ylabel("ERD/ERS (% Change from Baseline)")
    plt.title(f"ERD/ERS in {subject} ({condition}) [{fmin}-{fmax} Hz]")
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------- PLV AND COHERENCE FUNCTIONS  -------------------------------------------------------------------------------------

def plot_plv_matrix(plv_matrix, ch_names, title="PLV Matrix"):
    """
    Plot PLV matrix as a heatmap.

    Parameters:
        plv_matrix : np.ndarray
            PLV matrix (n_channels x n_channels).
        ch_names : list
            List of channel names (ordered as in the matrix).
        title : str
            Plot title.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(plv_matrix, interpolation="nearest", origin="lower", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.yticks(np.arange(len(ch_names)), ch_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_plv_difference(plv_real, plv_imagined, ch_names, title="PLV Difference (Real - Imagined)"):
    """
    Plot the difference between two PLV matrices as a heatmap.
    """
    diff = plv_real - plv_imagined

    plt.figure(figsize=(8, 6))
    im = plt.imshow(diff, interpolation="nearest", origin="lower", cmap="bwr", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.yticks(np.arange(len(ch_names)), ch_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_motor_plv_difference(plv_real, plv_imagined, ch_names, motor_chs=None, title="Motor Cortex PLV Difference (Real - Imagined)"):
    """
    Plot the PLV difference (real - imagined) for motor cortex channels only.
    """
    if motor_chs is None:
        motor_chs = ["C3", "C4", "C6", "CP5", "CP6", "FC3", "FC4", "Cz"]

    # Find indices of motor channels present in data
    indices = [ch_names.index(ch) for ch in motor_chs if ch in ch_names]
    motor_ch_names = [ch_names[i] for i in indices]

    # Extract sub-matrices for motor channels
    motor_plv_real = plv_real[np.ix_(indices, indices)]
    motor_plv_imagined = plv_imagined[np.ix_(indices, indices)]

    # Difference matrix
    diff = motor_plv_real - motor_plv_imagined

    # Plot
    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, interpolation="nearest", origin="lower", cmap="bwr", 
                    vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(motor_ch_names)), motor_ch_names, rotation=90)
    plt.yticks(np.arange(len(motor_ch_names)), motor_ch_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_plv_coherence(df, metric="PLV Mean"):
    """
    Plot PLV or Coherence comparisons between multiple conditions.

    Parameters:
    - df: DataFrame containing PLV & Coherence results
    - metric: "PLV Mean", "PLV Max", "Coherence Mean", or "Coherence Max"
    """
    plt.figure(figsize=(12, 6))

    # Automatically generate a color palette based on the number of unique conditions
    n_conditions = df["Condition"].nunique()
    palette = sns.color_palette("tab10", n_colors=n_conditions)

    sns.barplot(data=df, x="Channel Pair", y=metric, hue="Condition", palette=palette)

    plt.xlabel("Electrode Pair")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Across Conditions")
    plt.legend(title="Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
