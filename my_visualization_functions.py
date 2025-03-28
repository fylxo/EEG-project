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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import seaborn as sns
import mne



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




def visualize_comparison(group_results, individual_results):
    """
    Visualize and compare classification results between real vs. rest and real vs. imagined
    
    Parameters:
    -----------
    group_results : dict
        Dictionary containing group-level results from classify_condition_pairs
    individual_results : dict
        Dictionary containing individual subject results
    """
    # Check if we have both condition pairs
    pairs = ['real_right_hand_vs_rest', 'real_right_hand_vs_imagined_right_hand']
    if not all(pair in group_results for pair in pairs):
        raise ValueError("Both condition pairs must be in group_results")
    
    # Extract results
    real_vs_rest = group_results['real_right_hand_vs_rest']
    real_vs_imagined = group_results['real_right_hand_vs_imagined_right_hand']
    
    # 1. Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 2. Plot ROC curves
    ax1 = plt.subplot2grid((3, 3), (0, 0))

    # Real vs. Rest ROC
    ax1.plot(
        real_vs_rest['results']['fpr'], 
        real_vs_rest['results']['tpr'], 
        lw=2, 
        label=f"Real vs. Rest (AUC = {real_vs_rest['results']['roc_auc']:.2f})"
    )

    # Real vs. Imagined ROC
    ax1.plot(
        real_vs_imagined['results']['fpr'], 
        real_vs_imagined['results']['tpr'], 
        lw=2, 
        label=f"Real vs. Imagined (AUC = {real_vs_imagined['results']['roc_auc']:.2f})"
    )
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves Comparison')
    ax1.legend(loc="lower right")
    
    # 3. Plot accuracy comparison - BOTH group and individual averages
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    
    # Prepare data
    conditions = ['Real vs. Rest', 'Real vs. Imagined']
    
    # Group accuracies (subject-aware CV)
    group_accuracies = [
        real_vs_rest['results']['accuracy'],
        real_vs_imagined['results']['accuracy']
    # ]
    group_errors = [
        real_vs_rest['results']['std'],
        real_vs_imagined['results']['std']
    ]
    
    # Average of individual accuracies
    individual_accuracies = [
        real_vs_rest['individual_mean_acc'],
        real_vs_imagined['individual_mean_acc']
    ]
    individual_errors = [
        real_vs_rest['individual_std_acc'],
        real_vs_imagined['individual_std_acc']
    ]
    
    # Positions for bars
    x = np.arange(len(conditions))
    width = 0.35
    
    # Plot the bars
    ax2.bar(
        x - width/2, 
        group_accuracies, 
        width,
        yerr=group_errors, 
        capsize=10,
        color='#3498db',
        label='Group-Level (Subject-Aware CV)'
    )
    ax2.bar(
        x + width/2, 
        individual_accuracies, 
        width,
        yerr=individual_errors, 
        capsize=10,
        color='#e74c3c',
        label='Average of Individual Subjects'
    )
    
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy Comparison: Group vs. Individual Average')
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    
    # 4. Plot confusion matrices side by side

    # Real vs. Rest confusion matrix
    ax3 = plt.subplot2grid((3, 3), (1, 0))
    sns.heatmap(
        real_vs_rest['results']['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Rest', 'Real'],
        yticklabels=['Rest', 'Real'],
        ax=ax3
    )
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_title('Confusion Matrix - Real vs. Rest')
    
    # Real vs. Imagined confusion matrix
    ax4 = plt.subplot2grid((3, 3), (1, 1))
    sns.heatmap(
        real_vs_imagined['results']['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Reds',
        xticklabels=['Imagined', 'Real'],
        yticklabels=['Imagined', 'Real'],
        ax=ax4
    )
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.set_title('Confusion Matrix - Real vs. Imagined')
    
    # 5. Plot CSP patterns for real vs. rest
    ax5 = plt.subplot2grid((3, 3), (1, 2))
    try:
        # Get CSP patterns
        csp_real_rest = real_vs_rest['results']['pipeline'].named_steps['csp']
        info = real_vs_rest['info']
        channels = real_vs_rest['channels']
        
        # Create a copy of the info with only the selected channels
        info_reduced = mne.pick_info(
            info.copy(), 
            [info['ch_names'].index(ch) for ch in channels if ch in info['ch_names']]
        )
        
        # Plot patterns for Real vs. Rest
        pattern_data = csp_real_rest.patterns_[:, 0]
        mne.viz.plot_topomap(
            pattern_data, 
            info_reduced, 
            axes=ax5, 
            show=False
        )
        ax5.set_title('Most Discriminative Pattern\nReal vs. Rest')
    except Exception as e:
        print(f"Error plotting CSP patterns: {e}")
    
    # 6. Plot individual subject accuracies
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Get individual subejct accuracies
    subjects = sorted(individual_results['real_right_hand_vs_rest'].keys())
    real_rest_accs = [individual_results['real_right_hand_vs_rest'][subj]['accuracy'] for subj in subjects]
    real_imag_accs = [individual_results['real_right_hand_vs_imagined_right_hand'][subj]['accuracy'] 
                     for subj in subjects if subj in individual_results['real_right_hand_vs_imagined_right_hand']]
    
    # Ensure we have same number of subjects for both pairs
    if len(real_rest_accs) != len(real_imag_accs):
        min_len = min(len(real_rest_accs), len(real_imag_accs))
        subjects = subjects[:min_len]
        real_rest_accs = real_rest_accs[:min_len]
        real_imag_accs = real_imag_accs[:min_len]
    
    # Set up positions
    x = np.arange(len(subjects))
    width = 0.35
    
    # Plot bars
    ax6.bar(
        x - width/2, 
        real_rest_accs, 
        width,
        color='#3498db',
        label='Real vs. Rest'
    )
    ax6.bar(
        x + width/2, 
        real_imag_accs, 
        width,
        color='#e74c3c',
        label='Real vs. Imagined'
    )
    
    ax6.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
    ax6.set_ylabel('Classification Accuracy')
    ax6.set_xlabel('Subject')
    ax6.set_title('Individual Subject Classification Performance')
    ax6.set_xticks(x)
    ax6.set_xticklabels(subjects)
    ax6.legend()
    
    plt.tight_layout()
    plt.suptitle('Group-Level Classification Results Comparison', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Return summary text
    summary = (
        f"SUMMARY OF RESULTS:\n\n"
        f"Group-Level (Subject-Aware CV):\n"
        f"Real vs. Rest: Accuracy = {real_vs_rest['results']['accuracy']:.3f} ± {real_vs_rest['results']['std']:.3f}, "
        f"AUC = {real_vs_rest['results']['roc_auc']:.3f}\n"
        f"Real vs. Imagined: Accuracy = {real_vs_imagined['results']['accuracy']:.3f} ± {real_vs_imagined['results']['std']:.3f}, "
        f"AUC = {real_vs_imagined['results']['roc_auc']:.3f}\n\n"
        
        f"Average of Individual Subjects:\n"
        f"Real vs. Rest: Accuracy = {real_vs_rest['individual_mean_acc']:.3f} ± {real_vs_rest['individual_std_acc']:.3f}\n"
        f"Real vs. Imagined: Accuracy = {real_vs_imagined['individual_mean_acc']:.3f} ± {real_vs_imagined['individual_std_acc']:.3f}"
    )
    
    print("\n" + "="*80)
    print(summary)
    print("="*80)
    
    return summary