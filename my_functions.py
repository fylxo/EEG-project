"""
Functions for analyzing EEG data related to motor movement and imagery.
Includes preprocessing, connectivity analysis, and classification functions.
"""

# Core libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
import scipy.stats as stats
from scipy.signal import hilbert, coherence
from statsmodels.stats.multitest import multipletests

# MNE library for EEG processing
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from mne.decoding import CSP

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ------------------------------------------------------------------------------------- 
# PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------------- 

def process_eeg_automatic(subjects, data_path, apply_ica=False, bandpass=(6, 30), 
                         n_ica_components=25, tmin=-1.0, tmax=3.0):
    """
    Loads EEG data, applies preprocessing, and extracts epochs by condition.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to process
    data_path : str
        Path to the EEG data folder
    apply_ica : bool
        Whether to apply ICA for artifact removal
    bandpass : tuple
        Frequency band for filtering (low, high)
    n_ica_components : int
        Number of ICA components to extract
    tmin, tmax : float
        Epoch time range in seconds
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing preprocessed epochs by subject and condition
    """
    mne.set_log_level("ERROR")
    data_dict = {}

    # Define selected runs for each condition
    selected_runs = {
        "real_right_hand": [3, 7, 11],
        "imagined_right_hand": [4, 8, 12]
    }

    for subject in subjects:
        print(f"\n🔄 Processing {subject}...")
        data_dict[subject] = {"real_right_hand": [], "imagined_right_hand": [], "rest": []} 

        for condition, run_numbers in selected_runs.items():
            for run_number in run_numbers:
                run = f"R{str(run_number).zfill(2)}"
                edf_file = os.path.join(data_path, subject, f"{subject}{run}.edf")

                try:
                    # Load raw data
                    raw = mne.io.read_raw_edf(edf_file, preload=True)

                    # Standardize channel names and set montage
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)
                    raw.set_eeg_reference("average", projection=True)
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # Apply ICA for artifact removal if requested
                    if apply_ica:
                        raw_clean = _apply_ica(raw, n_ica_components, subject, run)
                    else:
                        raw_clean = raw

                    # Extract epochs for Right Hand and Rest conditions
                    events, event_id = mne.events_from_annotations(raw_clean)
                    if len(events) == 0:
                        print(f"⚠️ No events found for {subject} - {run}, skipping...")
                        continue

                    # Extract "Right Hand" epochs (T2)
                    if "T2" in event_id:
                        epochs = mne.Epochs(
                            raw_clean, events, event_id={"Right Hand": event_id["T2"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject][condition].append(epochs["Right Hand"])
                        print(f"✅ Extracted {len(epochs['Right Hand'])} epochs for {subject} - {run} ({condition})")

                    # Extract "Rest" epochs (T0)
                    if "T0" in event_id:
                        epochs_rest = mne.Epochs(
                            raw_clean, events, event_id={"Rest": event_id["T0"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject]["rest"].append(epochs_rest["Rest"])
                        print(f"✅ Extracted {len(epochs_rest['Rest'])} rest epochs for {subject} - {run}")

                except Exception as e:
                    print(f"⚠️ Skipping {edf_file} due to error: {e}")

    print("\n✅ Processing complete.")
    return data_dict


def process_eeg_manual(subjects, data_path, apply_ica=True, bandpass=(6, 30), 
                     n_ica_components=25, tmin=-1.0, tmax=2.0, manual_review=True):
    """
    Loads EEG data with manual ICA component review option.
    
    Similar to process_eeg_automatic, but allows for interactive
    review of ICA components for better artifact removal.
    """
    mne.set_log_level("ERROR")
    data_dict = {}

    selected_runs = {
        "real_right_hand": [3, 7, 11],
        "imagined_right_hand": [4, 8, 12]
    }

    for subject in subjects:
        print(f"\n🔄 Processing {subject}...")
        data_dict[subject] = {"real_right_hand": [], "imagined_right_hand": [], "rest": []}

        for condition, run_numbers in selected_runs.items():
            for run_number in run_numbers:
                run = f"R{str(run_number).zfill(2)}"
                edf_file = os.path.join(data_path, subject, f"{subject}{run}.edf")

                try:
                    # Load and preprocess raw data
                    raw = mne.io.read_raw_edf(edf_file, preload=True)
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)
                    raw.set_eeg_reference("average", projection=True)
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # Apply ICA with manual review
                    if apply_ica:
                        ica = _fit_ica(raw, n_ica_components, subject, run)
                        
                        # Manual review option
                        if manual_review:
                            print(f"🔎 Manual inspection time! Review ICA components for {subject} - {run}")
                            ica.plot_components(inst=raw)
                            plt.show()

                            if len(ica.exclude) > 0:
                                ica.plot_properties(raw, picks=ica.exclude)
                                plt.show()
                            
                            print(f"✅ Manual exclusions now: {ica.exclude}")

                        # Apply ICA
                        raw_clean = raw.copy()
                        ica.apply(raw_clean)
                    else:
                        raw_clean = raw

                    # Extract epochs
                    events, event_id = mne.events_from_annotations(raw_clean)
                    if len(events) == 0:
                        print(f"⚠️ No events found for {subject} - {run}, skipping...")
                        continue

                    # Right Hand epochs
                    if "T2" in event_id:
                        epochs = mne.Epochs(
                            raw_clean, events, event_id={"Right Hand": event_id["T2"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject][condition].append(epochs["Right Hand"])
                        print(f"✅ Extracted {len(epochs['Right Hand'])} epochs for {subject} - {run} ({condition})")

                    # Rest epochs
                    if "T0" in event_id:
                        epochs_rest = mne.Epochs(
                            raw_clean, events, event_id={"Rest": event_id["T0"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject]["rest"].append(epochs_rest["Rest"])
                        print(f"✅ Extracted {len(epochs_rest['Rest'])} rest epochs for {subject} - {run}")

                except Exception as e:
                    print(f"⚠️ Skipping {edf_file} due to error: {e}")

    print("\n✅ Processing complete.")
    return data_dict


def process_eeg_debug(subjects, data_path, bandpass=(6, 30), n_ica_components=25, 
                    tmin=-1.0, tmax=2.0):
    """
    Debugging version of the EEG processing pipeline with diagnostic plots.
    """
    mne.set_log_level("ERROR")
    data_dict = {}

    selected_runs = {
        "real_right_hand": [3, 7, 11],
        "imagined_right_hand": [4, 8, 12]
    }

    for subject in subjects:
        print(f"\n🔄 DEBUG Processing {subject}...")
        data_dict[subject] = {"real_right_hand": [], "imagined_right_hand": [], "rest": []}

        for condition, run_numbers in selected_runs.items():
            for run_number in run_numbers:
                run = f"R{str(run_number).zfill(2)}"
                edf_file = os.path.join(data_path, subject, f"{subject}{run}.edf")

                try:
                    # Load and preprocess data
                    raw = mne.io.read_raw_edf(edf_file, preload=True)
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)
                    raw.set_eeg_reference("average", projection=True)
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # Plot raw data before ICA
                    print(f"📊 Plotting RAW EEG before ICA - {subject} - {run}")
                    _plot_raw(raw, title=f"{subject} - {run} | Raw EEG BEFORE ICA")

                    # Fit ICA
                    ica = _fit_ica(raw, n_ica_components, subject, run)
                    
                    # Plot ICA components
                    print(f"🧠 Plotting ICA components for {subject} - {run}")
                    ica.plot_components(inst=raw)
                    plt.show()

                    # Plot excluded component properties
                    if len(ica.exclude) > 0:
                        print(f"📈 Plotting properties of excluded components")
                        ica.plot_properties(raw, picks=ica.exclude)
                        plt.show()

                    # Apply ICA
                    raw_clean = raw.copy()
                    ica.apply(raw_clean)

                    # Plot raw data after ICA
                    print(f"📊 Plotting RAW EEG after ICA - {subject} - {run}")
                    _plot_raw(raw_clean, title=f"{subject} - {run} | Raw EEG AFTER ICA")

                    # Extract epochs
                    events, event_id = mne.events_from_annotations(raw_clean)
                    if len(events) == 0:
                        print(f"⚠️ No events found for {subject} - {run}, skipping...")
                        continue

                    # Right Hand epochs
                    if "T2" in event_id:
                        epochs = mne.Epochs(
                            raw_clean, events, event_id={"Right Hand": event_id["T2"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject][condition].append(epochs["Right Hand"])
                        print(f"✅ Extracted {len(epochs['Right Hand'])} epochs for {subject} - {run} ({condition})")

                    # Rest epochs
                    if "T0" in event_id:
                        epochs_rest = mne.Epochs(
                            raw_clean, events, event_id={"Rest": event_id["T0"]},
                            tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
                        )
                        data_dict[subject]["rest"].append(epochs_rest["Rest"])
                        print(f"✅ Extracted {len(epochs_rest['Rest'])} rest epochs for {subject} - {run}")

                except Exception as e:
                    print(f"⚠️ Skipping {edf_file} due to error: {e}")

    print("\n✅ DEBUG processing complete.")
    return data_dict


def process_eeg(subjects, data_path, apply_ica=False, mode="automatic", **kwargs):
    """
    Wrapper for EEG processing pipeline with different modes.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    data_path : str
        Path to data directory
    apply_ica : bool
        Whether to apply ICA
    mode : str
        Processing mode: "automatic", "manual", or "debug"
    **kwargs : dict
        Additional arguments passed to specific processing functions
        
    Returns:
    --------
    data_dict : dict
        Dictionary of processed EEG data
    """
    if mode == "automatic":
        return process_eeg_automatic(subjects, data_path, apply_ica=apply_ica, **kwargs)
    elif mode == "manual":
        return process_eeg_manual(subjects, data_path, apply_ica=True, **kwargs)
    elif mode == "debug":
        return process_eeg_debug(subjects, data_path, **kwargs)
    else:
        raise ValueError("❌ Invalid mode! Choose from 'automatic', 'manual', or 'debug'.")


def compute_erd_ers(epochs, baseline=(-1, 0), fmin=6, fmax=30):
    """
    Compute Event-Related Desynchronization (ERD) & Synchronization (ERS).
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    baseline : tuple
        Time window for baseline (start, end) in seconds
    fmin, fmax : float
        Frequency range to analyze
        
    Returns:
    --------
    erd_ers : ndarray
        ERD/ERS values (epochs, channels, frequencies, times)
    times : ndarray
        Time points
    freqs : ndarray
        Frequency points
    """
    freqs = np.linspace(fmin, fmax, num=10)
    n_cycles = freqs / 2

    # Compute time-frequency power
    power = tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, 
        return_itc=False, decim=3, average=False
    )

    # Define baseline mask
    baseline_mask = (power.times >= baseline[0]) & (power.times <= baseline[1])

    # Compute baseline power (mean across time)
    baseline_power = np.mean(power.data[:, :, :, baseline_mask], axis=-1)

    # Compute ERD/ERS as percent change from baseline
    erd_ers = 100 * (power.data - baseline_power[:, :, :, np.newaxis]) / baseline_power[:, :, :, np.newaxis]
    
    return erd_ers, power.times, power.freqs


# Helper functions for ICA
def _fit_ica(raw, n_components, subject, run):
    """Helper function to fit ICA and detect artifacts"""
    ica = ICA(n_components=n_components, random_state=42, method="fastica", max_iter=500)
    ica.fit(raw)
    
    # EOG artifact detection
    eog_indices = []
    frontal_channels = [ch for ch in raw.ch_names 
                       if ch.lower().startswith(('fp', 'f')) 
                       and not ch.lower().startswith(('fc', 'ft'))][:2]
    
    if frontal_channels:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=frontal_channels)
    eog_indices = eog_indices[:2] if len(eog_indices) > 2 else eog_indices
    
    # ECG artifact detection using kurtosis
    ica_sources = ica.get_sources(raw).get_data()
    kurtosis_values = stats.kurtosis(ica_sources, axis=1)
    
    # Z-score kurtosis values
    kurt_z = (kurtosis_values - np.median(kurtosis_values)) / (
        np.median(np.abs(kurtosis_values - np.median(kurtosis_values))) + 1e-6
    )
    ecg_indices = np.where(kurt_z > 2.5)[0].tolist()[:2]
    
    # Muscle artifact detection
    muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.5, l_freq=7, h_freq=45)
    
    # Combine artifacts
    all_artifact_indices = list(set(eog_indices + ecg_indices + muscle_indices))
    ica.exclude = all_artifact_indices
    print(f"⚠️ Detected {len(ica.exclude)} artifact components: {ica.exclude}")
    
    return ica


def _apply_ica(raw, n_components, subject, run):
    """Apply ICA to raw data and return cleaned version"""
    ica = _fit_ica(raw, n_components, subject, run)
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean


def _plot_raw(raw, title="EEG Signal"):
    """Helper function to plot raw EEG data"""
    data, times = raw[:, :]
    plt.figure(figsize=(12, 6))
    for i in range(min(10, data.shape[0])):
        plt.plot(times, data[i] * 1e6 + i * 100, label=raw.ch_names[i])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (µV)")
    plt.legend(loc="upper right")
    plt.show()


# ------------------------------------------------------------------------------------- 
# PLV AND COHERENCE FUNCTIONS
# ------------------------------------------------------------------------------------- 

def compute_plv_matrix(epochs):
    """
    Compute full PLV matrix across all channels in an Epochs object.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
        
    Returns:
    --------
    plv_matrix : ndarray
        Phase Locking Value matrix (n_channels x n_channels)
    """
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    plv_accum = np.zeros((n_channels, n_channels))

    for epoch_idx in range(n_epochs):
        epoch_data = data[epoch_idx]
        analytic_signal = hilbert(epoch_data, axis=1)
        phase_data = np.angle(analytic_signal)

        plv_epoch = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                phase_diff = phase_data[i] - phase_data[j]
                plv_epoch[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))

        plv_accum += plv_epoch

    plv_matrix = plv_accum / n_epochs
    return plv_matrix


def compute_plv_pairwise(epochs, ch1="C3", ch2="C4"):
    """
    Compute PLV between two channels across all epochs.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    ch1, ch2 : str
        Channel names to compute PLV between
        
    Returns:
    --------
    plv : ndarray
        PLV time series
    plv_mean : float
        Mean PLV value
    plv_max : float
        Maximum PLV value
    """
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    data = epochs.get_data(picks=[ch1, ch2])

    if data.shape[0] == 0:
        return np.nan, np.nan, np.nan

    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    phase_diff = np.exp(1j * (phase_data[:, 0, :] - phase_data[:, 1, :]))
    plv = np.abs(np.mean(phase_diff, axis=0))

    plv_mean = np.mean(plv)
    plv_max = np.max(plv)

    return plv, plv_mean, plv_max


def compute_coherence(epochs, ch1="C3", ch2="C4", fmin=8, fmax=30, nperseg=256):
    """
    Compute coherence between two EEG channels using Welch's method.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    ch1, ch2 : str
        Channel names
    fmin, fmax : float
        Frequency range
    nperseg : int
        Length of each segment for Welch's method
        
    Returns:
    --------
    freqs : ndarray
        Frequency vector
    coherence_values : ndarray
        Coherence spectrum in the selected band
    coherence_mean : float
        Mean coherence in the frequency range
    coherence_max : float
        Max coherence in the frequency range
    """
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    sfreq = epochs.info['sfreq']
    ch1_idx = epochs.ch_names.index(ch1)
    ch2_idx = epochs.ch_names.index(ch2)

    data = epochs.get_data(picks=[ch1_idx, ch2_idx])
    data_avg = np.mean(data, axis=0)

    freqs, coh_values = coherence(data_avg[0, :], data_avg[1, :], fs=sfreq, nperseg=nperseg)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    selected_frequencies = freqs[freq_mask]
    selected_coherence = coh_values[freq_mask]

    coherence_mean = np.mean(selected_coherence)
    coherence_max = np.max(selected_coherence)

    return selected_frequencies, selected_coherence, coherence_mean, coherence_max


def analyze_pairwise_plv_coherence(subjects, eeg_data, conditions, channel_pairs, fmin=8, fmax=30):
    """
    Computes PLV and Coherence for multiple subject-channel pairs.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    eeg_data : dict
        Dictionary with subject-level preprocessed data
    conditions : dict
        Dictionary mapping condition names to keys in eeg_data
    channel_pairs : list
        List of tuples with channel pairs (e.g., [("C3", "C4")])
    fmin, fmax : float
        Frequency band limits
        
    Returns:
    --------
    df : DataFrame
        DataFrame with PLV & coherence values for all pairs, subjects & conditions
    """
    results = []

    for subject in subjects:
        print(f"🧠 Subject: {subject}")

        for cond_key, cond_value in conditions.items():
            print(f"  ➡️ Condition: {cond_key}")

            # Skip if subject or condition not available
            if subject not in eeg_data or not eeg_data[subject].get(cond_value):
                print(f"  ⚠️ No data for {subject} - {cond_value}")
                continue
                
            # Merge runs
            epochs = mne.concatenate_epochs(eeg_data[subject][cond_value])

            for ch1, ch2 in channel_pairs:
                try:
                    # Compute PLV
                    plv, plv_mean, plv_max = compute_plv_pairwise(epochs, ch1=ch1, ch2=ch2)

                    # Compute Coherence
                    _, coh_spectrum, coh_mean, coh_max = compute_coherence(
                        epochs, ch1=ch1, ch2=ch2, fmin=fmin, fmax=fmax
                    )

                    results.append({
                        "Subject": subject,
                        "Condition": cond_key.capitalize(),
                        "Channel Pair": f"{ch1}-{ch2}",
                        "PLV Mean": plv_mean,
                        "PLV Max": plv_max,
                        "Coherence Mean": coh_mean,
                        "Coherence Max": coh_max
                    })
                except Exception as e:
                    print(f"  ⚠️ Error processing {ch1}-{ch2}: {e}")

    df = pd.DataFrame(results)
    return df


def plot_normalized_plv_coherence(df, metric="PLV Mean", figsize=(14, 7)):
    """
    Plot PLV or Coherence normalized to the Rest condition.
    Rest condition is not shown but serves as the baseline (value 1.0).
    
    Parameters:
    -----------
    df : DataFrame
        Original PLV/Coherence DataFrame
    metric : str
        Metric to normalize: e.g., "PLV Mean"
    figsize : tuple
        Size of the matplotlib figure
    """
    if metric not in ["PLV Mean", "PLV Max", "Coherence Mean", "Coherence Max"]:
        raise ValueError("Invalid metric.")

    df_pivot = df.pivot_table(index=["Subject", "Channel Pair"], columns="Condition", values=metric)
    df_pivot = df_pivot.dropna(subset=["Rest"])

    # Normalize Real and Imagined
    df_norm = df_pivot.copy()
    df_norm["Real"] = df_norm["Real"] / df_norm["Rest"]
    df_norm["Imagined"] = df_norm["Imagined"] / df_norm["Rest"]

    df_long = df_norm[["Real", "Imagined"]].reset_index().melt(
        id_vars=["Subject", "Channel Pair"],
        var_name="Condition",
        value_name=metric
    )

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_long, x="Channel Pair", y=metric, hue="Condition", palette="Set2")

    # Reference line for Rest (value = 1)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label="Rest Baseline")

    # Labels
    plt.xlabel("Electrode Pair")
    plt.ylabel(f"Normalized {metric} (÷ Rest)")
    plt.title(f"Normalized {metric} (Real/Imagined divided by Rest)")
    plt.xticks(rotation=45)
    plt.legend(title="Condition")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def paired_ttest_plv(df, metric="PLV Mean", alpha=0.05):
    """
    Run paired t-tests for PLV between conditions on each channel pair.
    Applies Bonferroni correction for multiple comparisons.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with PLV results from analyze_pairwise_plv_coherence
    metric : str
        Which metric to use for t-tests
    alpha : float
        Significance level before correction
        
    Returns:
    --------
    df_stats : DataFrame
        DataFrame with t-test results
    """
    from scipy.stats import ttest_rel
    
    pairs = df["Channel Pair"].unique()
    results = []

    # Step 1: Collect all p-values for both comparisons
    p_values_real_rest = []
    p_values_real_imagined = []

    for pair in pairs:
        df_pair = df[df["Channel Pair"] == pair]
        pivot_df = df_pair.pivot(index="Subject", columns="Condition", values=metric)
        pivot_df = pivot_df.dropna()

        # Only perform test if we have both conditions for at least 2 subjects
        if ("Real" in pivot_df.columns and "Rest" in pivot_df.columns and 
            len(pivot_df) >= 2):
            t_real_rest, p_real_rest = ttest_rel(pivot_df["Real"], pivot_df["Rest"])
            p_values_real_rest.append(p_real_rest)
        else:
            t_real_rest, p_real_rest = np.nan, np.nan
            
        if ("Real" in pivot_df.columns and "Imagined" in pivot_df.columns and 
            len(pivot_df) >= 2):
            t_real_imagined, p_real_imagined = ttest_rel(pivot_df["Real"], pivot_df["Imagined"])
            p_values_real_imagined.append(p_real_imagined)
        else:
            t_real_imagined, p_real_imagined = np.nan, np.nan

        results.append({
            "Channel Pair": pair,
            "t Real vs Rest": t_real_rest,
            "p Real vs Rest": p_real_rest,
            "t Real vs Imagined": t_real_imagined,
            "p Real vs Imagined": p_real_imagined
        })

    df_stats = pd.DataFrame(results)
    return df_stats


def report_paired_ttests(df_stats):
    """
    Display t-test results for PLV in a clear and readable way.
    
    Parameters:
    -----------
    df_stats : DataFrame
        DataFrame with t-test results from paired_ttest_plv
    """
    for idx, row in df_stats.iterrows():
        print(f"\n📊 Channel Pair: {row['Channel Pair']}")

        # Real vs Rest
        p1 = row['p Real vs Rest']
        if not np.isnan(p1):
            color1 = "\033[92m" if p1 <= 0.05 else "\033[90m"
            print(f"{color1}   Real vs Rest: t = {row['t Real vs Rest']:.2f}, p = {p1:.4f} \033[0m")
            
            if 'p Real vs Rest (Bonf)' in row:
                p1_corr = row['p Real vs Rest (Bonf)']
                sig = "significant" if p1_corr <= 0.05 else "not significant"
                print(f"     After Bonferroni correction: p = {p1_corr:.4f} ({sig})")
        else:
            print("   Real vs Rest: Insufficient data for comparison")

        # Real vs Imagined
        p2 = row['p Real vs Imagined']
        if not np.isnan(p2):
            color2 = "\033[92m" if p2 <= 0.05 else "\033[90m"
            print(f"{color2}   Real vs Imagined: t = {row['t Real vs Imagined']:.2f}, p = {p2:.4f} \033[0m")
            
            if 'p Real vs Imagined (Bonf)' in row:
                p2_corr = row['p Real vs Imagined (Bonf)']
                sig = "significant" if p2_corr <= 0.05 else "not significant"
                print(f"     After Bonferroni correction: p = {p2_corr:.4f} ({sig})")
        else:
            print("   Real vs Imagined: Insufficient data for comparison")


def plot_ttest_summary(df_stats, alpha=0.05, figsize=(10, 6)):
    """
    Graphical summary of t-test p-values with significance indication.
    
    Parameters:
    -----------
    df_stats : DataFrame
        DataFrame from paired_ttest_plv
    alpha : float
        Significance threshold
    figsize : tuple
        Size of the figure
    """
    import matplotlib.colors as mcolors

    pval_df = df_stats[["Channel Pair", "p Real vs Rest", "p Real vs Imagined"]].copy()
    pval_df.set_index("Channel Pair", inplace=True)

    # Replace missing with 1 (non-significant)
    pval_matrix = pval_df.fillna(1.0)

    # Create mask for significance
    sig_mask = pval_matrix <= alpha

    # Custom colormap: light = non-sig, dark = sig
    cmap = sns.light_palette("crimson", as_cmap=True)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pval_matrix,
        cmap=cmap,
        annot=pval_matrix.applymap(lambda x: f"{x:.4f}"),
        fmt="",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "p-value"},
        vmin=0, vmax=1
    )

    # Overlay asterisk for significant results
    for y in range(pval_matrix.shape[0]):
        for x in range(pval_matrix.shape[1]):
            if sig_mask.iloc[y, x]:
                ax.text(x + 0.5, y + 0.5, "*", ha='center', va='center', fontsize=18, color='black')

    ax.set_title(f"T-test P-Values (Significant p ≤ {alpha})\n* = statistically significant")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Channel Pair")
    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------------------------------- 
# iPLV AND wPLI FUNCTIONS
# ------------------------------------------------------------------------------------- 

def compute_iplv_matrix(epochs):
    """
    Compute imaginary PLV matrix across all channels in an Epochs object.
    iPLV only considers the imaginary part of the phase difference, 
    making it less susceptible to volume conduction.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
        
    Returns:
    --------
    iplv_matrix : ndarray
        Imaginary Phase Locking Value matrix (n_channels x n_channels)
    """
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    iplv_accum = np.zeros((n_channels, n_channels))

    for epoch_idx in range(n_epochs):
        epoch_data = data[epoch_idx]
        analytic_signal = hilbert(epoch_data, axis=1)
        phase_data = np.angle(analytic_signal)

        iplv_epoch = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                phase_diff = phase_data[i] - phase_data[j]
                # Take absolute value of the imaginary part only
                iplv_epoch[i, j] = np.abs(np.mean(np.sin(phase_diff)))

        iplv_accum += iplv_epoch

    iplv_matrix = iplv_accum / n_epochs
    return iplv_matrix


def compute_wpli_matrix(epochs):
    """
    Compute weighted Phase Lag Index (wPLI) matrix across all channels.
    wPLI is robust against volume conduction and common reference effects.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
        
    Returns:
    --------
    wpli_matrix : ndarray
        Weighted Phase Lag Index matrix (n_channels x n_channels)
    """
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    
    # Initialize cross-spectral density matrices
    imag_sum = np.zeros((n_channels, n_channels))
    imag_abs_sum = np.zeros((n_channels, n_channels))
    
    for epoch_idx in range(n_epochs):
        epoch_data = data[epoch_idx]
        analytic_signal = hilbert(epoch_data, axis=1)
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):  # Upper triangle only
                cross_spectrum = analytic_signal[i] * np.conj(analytic_signal[j])
                imag_part = np.imag(cross_spectrum)
                
                # Accumulate for wPLI computation
                imag_sum[i, j] += np.mean(imag_part)
                imag_abs_sum[i, j] += np.mean(np.abs(imag_part))
    
    # Compute wPLI in upper triangle
    wpli_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if imag_abs_sum[i, j] > 0:
                wpli_matrix[i, j] = np.abs(imag_sum[i, j]) / imag_abs_sum[i, j]
    
    # Make symmetric
    wpli_matrix = wpli_matrix + wpli_matrix.T
    
    return wpli_matrix


def compute_iplv_pairwise(epochs, ch1="C3", ch2="C4"):
    """
    Compute imaginary PLV between two channels across all epochs.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    ch1, ch2 : str
        Channel names to compute iPLV between
        
    Returns:
    --------
    iplv : ndarray
        iPLV time series
    iplv_mean : float
        Mean iPLV value
    iplv_max : float
        Maximum iPLV value
    """
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    data = epochs.get_data(picks=[ch1, ch2])

    if data.shape[0] == 0:
        return np.nan, np.nan, np.nan

    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    phase_diff = phase_data[:, 0, :] - phase_data[:, 1, :]
    iplv = np.abs(np.mean(np.sin(phase_diff), axis=0))

    iplv_mean = np.mean(iplv)
    iplv_max = np.max(iplv)

    return iplv, iplv_mean, iplv_max


def compute_wpli_pairwise(epochs, ch1="C3", ch2="C4"):
    """
    Compute weighted Phase Lag Index (wPLI) between two channels.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    ch1, ch2 : str
        Channel names to compute wPLI between
        
    Returns:
    --------
    wpli : float
        wPLI value
    """
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    data = epochs.get_data(picks=[ch1, ch2])

    if data.shape[0] == 0:
        return np.nan, np.nan

    # Compute analytic signal
    analytic_signal = hilbert(data)
    
    # Get signals for each channel
    ch1_signal = analytic_signal[:, 0, :]
    ch2_signal = analytic_signal[:, 1, :]
    
    # Compute cross-spectrum for each epoch
    imag_sum = 0
    imag_abs_sum = 0
    
    for epoch_idx in range(data.shape[0]):
        cross_spectrum = ch1_signal[epoch_idx] * np.conj(ch2_signal[epoch_idx])
        imag_part = np.imag(cross_spectrum)
        
        imag_sum += np.mean(imag_part)
        imag_abs_sum += np.mean(np.abs(imag_part))
    
    # Calculate wPLI
    wpli = np.abs(imag_sum) / (imag_abs_sum + 1e-10)  # Add small value to avoid division by zero
    
    return wpli, wpli  # Return twice for consistency with other functions (mean/max)

def analyze_pairwise_connectivity(subjects, eeg_data, conditions, channel_pairs, metrics=["iPLV", "wPLI"]):
    """
    Computes iPLV and wPLI for multiple subject-channel pairs.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    eeg_data : dict
        Dictionary with subject-level preprocessed data
    conditions : dict
        Dictionary mapping condition names to keys in eeg_data
    channel_pairs : list
        List of tuples with channel pairs (e.g., [("C3", "C4")])
    metrics : list
        List of metrics to compute (default = ["iPLV", "wPLI"])
        
    Returns:
    --------
    df : DataFrame
        DataFrame with connectivity values for all pairs, subjects & conditions
    """
    results = []

    for subject in subjects:
        print(f"🧠 Subject: {subject}")

        for cond_key, cond_value in conditions.items():
            print(f"  ➡️ Condition: {cond_key}")

            # Skip if subject or condition not available
            if subject not in eeg_data or not eeg_data[subject].get(cond_value):
                print(f"  ⚠️ No data for {subject} - {cond_value}")
                continue
                
            # Merge runs
            epochs = mne.concatenate_epochs(eeg_data[subject][cond_value])

            for ch1, ch2 in channel_pairs:
                try:
                    result_row = {
                        "Subject": subject,
                        "Condition": cond_key.capitalize(),
                        "Channel Pair": f"{ch1}-{ch2}"
                    }
                    
                    # Compute iPLV if requested
                    if "iPLV" in metrics:
                        _, iplv_mean, iplv_max = compute_iplv_pairwise(epochs, ch1=ch1, ch2=ch2)
                        result_row["iPLV Mean"] = iplv_mean
                        result_row["iPLV Max"] = iplv_max
                        
                    # Compute wPLI if requested
                    if "wPLI" in metrics:
                        wpli_value, _ = compute_wpli_pairwise(epochs, ch1=ch1, ch2=ch2)
                        result_row["wPLI"] = wpli_value
                        
                    # Add regular coherence for comparison
                    if "Coherence" in metrics:
                        _, coh_spectrum, coh_mean, coh_max = compute_coherence(
                            epochs, ch1=ch1, ch2=ch2, fmin=8, fmax=30
                        )
                        result_row["Coherence Mean"] = coh_mean
                        result_row["Coherence Max"] = coh_max

                    results.append(result_row)
                except Exception as e:
                    print(f"  ⚠️ Error processing {ch1}-{ch2}: {e}")

    df = pd.DataFrame(results)
    return df


def paired_ttest_connectivity(df, metric="iPLV Mean", alpha=0.05):
    """
    Run paired t-tests for connectivity metrics between conditions on each channel pair.
    Applies Bonferroni correction for multiple comparisons.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with connectivity results
    metric : str
        Which metric to use for t-tests (e.g., "iPLV Mean", "wPLI")
    alpha : float
        Significance level before correction
        
    Returns:
    --------
    df_stats : DataFrame
        DataFrame with t-test results
    """
    from scipy.stats import ttest_rel
    
    pairs = df["Channel Pair"].unique()
    results = []

    # Step 1: Collect all p-values for both comparisons
    p_values_real_rest = []
    p_values_real_imagined = []

    for pair in pairs:
        df_pair = df[df["Channel Pair"] == pair]
        pivot_df = df_pair.pivot(index="Subject", columns="Condition", values=metric)
        pivot_df = pivot_df.dropna()

        # Only perform test if we have both conditions for at least 2 subjects
        if ("Real" in pivot_df.columns and "Rest" in pivot_df.columns and 
            len(pivot_df) >= 2):
            t_real_rest, p_real_rest = ttest_rel(pivot_df["Real"], pivot_df["Rest"])
            p_values_real_rest.append(p_real_rest)
        else:
            t_real_rest, p_real_rest = np.nan, np.nan
            
        if ("Real" in pivot_df.columns and "Imagined" in pivot_df.columns and 
            len(pivot_df) >= 2):
            t_real_imagined, p_real_imagined = ttest_rel(pivot_df["Real"], pivot_df["Imagined"])
            p_values_real_imagined.append(p_real_imagined)
        else:
            t_real_imagined, p_real_imagined = np.nan, np.nan

        results.append({
            "Channel Pair": pair,
            "t Real vs Rest": t_real_rest,
            "p Real vs Rest": p_real_rest,
            "t Real vs Imagined": t_real_imagined,
            "p Real vs Imagined": p_real_imagined
        })

    # Apply Bonferroni correction
    p_values_real_rest = [p for p in p_values_real_rest if not np.isnan(p)]
    p_values_real_imagined = [p for p in p_values_real_imagined if not np.isnan(p)]
    
    if p_values_real_rest:
        _, p_corrected_real_rest, _, _ = multipletests(
            p_values_real_rest, alpha=alpha, method='bonferroni')
    
    if p_values_real_imagined:
        _, p_corrected_real_imagined, _, _ = multipletests(
            p_values_real_imagined, alpha=alpha, method='bonferroni')
    
    # Add corrected p-values to results
    p_rest_idx = 0
    p_imag_idx = 0
    
    for i, result in enumerate(results):
        if not np.isnan(result["p Real vs Rest"]) and p_values_real_rest:
            results[i]["p Real vs Rest (Bonf)"] = p_corrected_real_rest[p_rest_idx]
            p_rest_idx += 1
            
        if not np.isnan(result["p Real vs Imagined"]) and p_values_real_imagined:
            results[i]["p Real vs Imagined (Bonf)"] = p_corrected_real_imagined[p_imag_idx]
            p_imag_idx += 1

    df_stats = pd.DataFrame(results)
    return df_stats


def plot_iplv_wpli_comparison(df, metrics=["iPLV Mean", "wPLI"], figsize=(14, 8)):
    """
    Plot comparison of different connectivity metrics across conditions.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with connectivity metrics
    metrics : list
        List of metrics to include in the plot
    figsize : tuple
        Figure size
    """
    import seaborn as sns
    
    # Reshape data for plotting
    plot_data = []
    
    for metric in metrics:
        if metric in df.columns:
            temp_df = df[["Subject", "Condition", "Channel Pair", metric]].copy()
            temp_df["Metric"] = metric
            temp_df["Value"] = temp_df[metric]
            plot_data.append(temp_df[["Subject", "Condition", "Channel Pair", "Metric", "Value"]])
    
    if not plot_data:
        raise ValueError("No valid metrics found in DataFrame")
        
    plot_df = pd.concat(plot_data)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        sns.boxplot(data=plot_df[plot_df["Metric"] == metric], 
                   x="Channel Pair", y="Value", hue="Condition",
                   palette="Set2")
        
        plt.title(f"{metric} by Channel Pair")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Only show legend for the first plot to save space
        if i > 0:
            plt.legend([])
    
    plt.tight_layout()
    plt.show()

def plot_connectivity_barplots(df, metrics=["iPLV Mean", "wPLI"], figsize=(12, 6), 
                               palette="Set2", error_bars=True, save_path=None):
    """
    Create separate bar plots for each connectivity metric.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with connectivity metrics
    metrics : list
        List of metrics to plot
    figsize : tuple
        Size for each individual figure
    palette : str or list
        Color palette for the bars
    error_bars : bool
        Whether to include error bars (standard error)
    save_path : str or None
        Path to save the figures (None = don't save)
        Will save as {save_path}_{metric}.png
    
    Returns:
    --------
    fig_dict : dict
        Dictionary containing the created figures
    """
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    
    fig_dict = {}  # Store created figures
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame, skipping")
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig_dict[metric] = fig
        
        # Calculate means and standard errors for cleaner bar plots
        summary_df = df.groupby(['Channel Pair', 'Condition'])[metric].agg(['mean', 'std', 'count']).reset_index()
        summary_df['se'] = summary_df['std'] / np.sqrt(summary_df['count'])
        
        # Create bar plot
        if error_bars:
            # With error bars
            sns.barplot(
                data=summary_df,
                x='Channel Pair',
                y='mean',
                hue='Condition',
                palette=palette,
                errorbar=('se', 1),  # Show standard error
                ax=ax
            )
        else:
            # Without error bars
            sns.barplot(
                data=summary_df,
                x='Channel Pair',
                y='mean',
                hue='Condition',
                palette=palette,
                errorbar=None,
                ax=ax
            )
        
        # Enhance the plot
        ax.set_title(f"{metric} by Channel Pair", fontsize=14, pad=20)
        ax.set_xlabel("Electrode Pair", fontsize=12, labelpad=10)
        ax.set_ylabel(f"{metric}", fontsize=12, labelpad=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Make y-axis start at 0
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Add value labels on top of bars if there aren't too many bars
        if len(summary_df) <= 20:  # Only add labels if there aren't too many bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=8, color='black',
                    xytext=(0, 3),
                    textcoords='offset points'
                )
        
        # Enhance legend
        ax.legend(
            title="Condition", 
            title_fontsize=12,
            frameon=True, 
            framealpha=0.9,
            edgecolor='gray'
        )
        
        # Tight layout to avoid clipping
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Replace spaces and special characters
            metric_filename = metric.replace(" ", "_").replace("/", "_")
            plt.savefig(f"{save_path}_{metric_filename}.png", dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}_{metric_filename}.png")
        
        # Show the plot
        plt.show()
    
    return fig_dict




def plot_connectivity_matrix(matrix, ch_names, title="Connectivity Matrix", cmap="viridis", vmin=None, vmax=None):
    """
    Plot connectivity matrix (iPLV, wPLI, etc.) as heatmap.
    
    Parameters:
    -----------
    matrix : ndarray
        Square connectivity matrix
    ch_names : list
        Channel names
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float or None
        Min and max values for color scaling
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Connectivity Strength")
    
    # Add channel labels
    plt.xticks(range(len(ch_names)), ch_names, rotation=90)
    plt.yticks(range(len(ch_names)), ch_names)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_connectivity_difference(matrix1, matrix2, ch_names, 
                               title="Connectivity Difference", cmap="coolwarm"):
    """
    Plot difference between two connectivity matrices.
    
    Parameters:
    -----------
    matrix1, matrix2 : ndarray
        Square connectivity matrices
    ch_names : list
        Channel names
    title : str
        Plot title
    cmap : str
        Colormap name (default coolwarm for difference plots)
    """
    diff_matrix = matrix1 - matrix2
    
    # Determine symmetric scale for better visualization
    maxval = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(diff_matrix, cmap=cmap, interpolation="nearest", origin="lower", vmin=-maxval, vmax=maxval)
    plt.colorbar(im, label="Connectivity Difference")
    
    # Add channel labels
    plt.xticks(range(len(ch_names)), ch_names, rotation=90)
    plt.yticks(range(len(ch_names)), ch_names)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()




def plot_normalized_connectivity(df, metrics=["iPLV Mean", "wPLI"], figsize=(14, 7)):
    """
    Plot connectivity metrics normalized to the Rest condition.
    Rest condition is not shown but serves as the baseline (value 1.0).
    
    Parameters:
    -----------
    df : DataFrame
        Original connectivity DataFrame
    metrics : list
        Metrics to normalize and plot
    figsize : tuple
        Size of the matplotlib figure
    """
    import seaborn as sns
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame, skipping")
            continue
            
        print(f"\nPlotting normalized {metric}:")
            
        df_pivot = df.pivot_table(index=["Subject", "Channel Pair"], columns="Condition", values=metric)
        df_pivot = df_pivot.dropna(subset=["Rest"])

        # Normalize Real and Imagined
        df_norm = df_pivot.copy()
        df_norm["Real"] = df_norm["Real"] / df_norm["Rest"]
        df_norm["Imagined"] = df_norm["Imagined"] / df_norm["Rest"]

        df_long = df_norm[["Real", "Imagined"]].reset_index().melt(
            id_vars=["Subject", "Channel Pair"],
            var_name="Condition",
            value_name=metric
        )

        # Plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df_long, x="Channel Pair", y=metric, hue="Condition", palette="Set2")

        # Reference line for Rest (value = 1)
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label="Rest Baseline")

        # Labels
        plt.xlabel("Electrode Pair")
        plt.ylabel(f"Normalized {metric} (÷ Rest)")
        plt.title(f"Normalized {metric} (Real/Imagined divided by Rest)")
        plt.xticks(rotation=45)
        plt.legend(title="Condition")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()





# ------------------------------------------------------------------------------------- 
# DECODING PIPELINE
# ------------------------------------------------------------------------------------- 

def perform_group_csp_lda(X, y, groups, title="Group Classification"):
    """
    Perform CSP+LDA classification with subject-aware cross-validation.
    
    Parameters:
    -----------
    X : ndarray
        Input features of shape (n_samples, n_channels, n_times)
    y : ndarray
        Target labels
    groups : ndarray
        Group labels for cross-validation (e.g., subject IDs)
    title : str
        Title for printing results
        
    Returns:
    --------
    results : dict
        Dictionary with classification results
    """
    # Define group-stratified cross-validation
    cv = GroupKFold(n_splits=5)
    
    # Define pipeline
    pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True)),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    
    # Cross-validation
    scores = []
    y_pred = np.zeros_like(y)
    y_prob = np.zeros_like(y, dtype=float)
    
    for train_idx, test_idx in cv.split(X, y, groups):
        pipeline.fit(X[train_idx], y[train_idx])
        
        # Predict
        y_pred[test_idx] = pipeline.predict(X[test_idx])
        
        try:
            y_prob[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
        except:
            y_prob[test_idx] = pipeline.decision_function(X[test_idx])
            
        # Calculate fold accuracy
        fold_acc = np.mean(y_pred[test_idx] == y[test_idx])
        scores.append(fold_acc)
    
    # Overall metrics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"{title}: Accuracy = {mean_score:.3f} ± {std_score:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # ROC curve metrics
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Train on all data to get CSP patterns
    pipeline.fit(X, y)
    
    return {
        'accuracy': mean_score,
        'std': std_score,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'pipeline': pipeline
    }


def perform_csp_lda(X, y, title="Classification"):
    """
    Perform CSP+LDA classification with stratified cross-validation.
    
    Parameters:
    -----------
    X : ndarray
        Input features of shape (n_samples, n_channels, n_times)
    y : ndarray
        Target labels
    title : str
        Title for printing results
        
    Returns:
    --------
    results : dict
        Dictionary with classification results
    """
    # Define cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define pipeline
    pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True)),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"{title}: Accuracy = {mean_score:.3f} ± {std_score:.3f}")
    
    # Predictions for confusion matrix and ROC curve
    y_pred = np.zeros_like(y)
    y_prob = np.zeros_like(y, dtype=float)
    
    for train_idx, test_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipeline.predict(X[test_idx])
        
        try:
            y_prob[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
        except:
            y_prob[test_idx] = pipeline.decision_function(X[test_idx])
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # ROC curve metrics
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Train on all data to get CSP patterns
    pipeline.fit(X, y)
    
    return {
        'accuracy': mean_score,
        'std': std_score,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'pipeline': pipeline
    }


def classify_condition_pairs(eeg_data, subjects, freq_band=(8, 30)):
    """
    Perform CSP+LDA classification on multiple condition pairs and return group results.
    
    Parameters:
    -----------
    eeg_data : dict
        Dictionary of EEG data organized by subject and condition
    subjects : list
        List of subject IDs
    freq_band : tuple
        Frequency band for filtering
        
    Returns:
    --------
    group_results : dict
        Dictionary containing group-level results for different condition pairs
    all_individual_results : dict
        Dictionary containing all individual subject results
    """
    # Define condition pairs to analyze
    condition_pairs = [
        ('real_right_hand', 'rest'),
        ('real_right_hand', 'imagined_right_hand')
    ]
    
    # Results dictionary for group analysis
    group_results = {}
    all_individual_results = {}
    
    # Process each condition pair
    for condition_pair in condition_pairs:
        condition1, condition2 = condition_pair
        pair_name = f"{condition1}_vs_{condition2}"
        print(f"\n{'='*80}\nAnalyzing {condition1} vs {condition2}\n{'='*80}")
        
        # Process all subjects for this condition pair
        X_all = []
        y_all = []
        subject_ids = []
        individual_results = {}
        
        for subject in subjects:
            print(f"\nProcessing subject: {subject}")
            
            try:
                # Check if required conditions exist
                if not (condition1 in eeg_data[subject] and condition2 in eeg_data[subject]):
                    print(f"Skipping subject {subject}: missing condition data")
                    continue
                
                # Skip if no epochs available
                if not eeg_data[subject][condition1] or not eeg_data[subject][condition2]:
                    print(f"Skipping subject {subject}: empty epoch list")
                    continue
                    
                # Concatenate epochs for this subject
                epochs1 = mne.concatenate_epochs(eeg_data[subject][condition1])
                epochs2 = mne.concatenate_epochs(eeg_data[subject][condition2])
                
                # Apply bandpass filter
                fmin, fmax = freq_band
                epochs1 = epochs1.copy().filter(fmin, fmax)
                epochs2 = epochs2.copy().filter(fmin, fmax)
                
                # Select motor channels
                motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'FC1', 'FC2', 'CP1', 'CP2']
                available_channels = [ch for ch in motor_channels if ch in epochs1.ch_names]
                
                if not available_channels:
                    print(f"Skipping subject {subject}: no motor channels available")
                    continue
                
                # Pick channels and extract time window of interest (0-2s)
                epochs1_motor = epochs1.copy().pick_channels(available_channels)
                epochs2_motor = epochs2.copy().pick_channels(available_channels)
                
                # Get time indices
                times = epochs1_motor.times
                start_idx = np.where(times >= 0)[0][0]
                end_idx = np.where(times >= 2.0)[0][0] if any(times >= 2.0) else -1
                
                # Extract data
                X1 = epochs1_motor.get_data()[:, :, start_idx:end_idx]  # condition1
                X2 = epochs2_motor.get_data()[:, :, start_idx:end_idx]  # condition2
                
                # Balance classes through random undersampling
                if len(X2) > len(X1):
                    indices = np.random.choice(len(X2), len(X1), replace=False)
                    X2 = X2[indices]
                    print(f"Subsampled {condition2} to match {condition1}: {len(X1)} trials")
                
                # Create combined dataset for this subject
                X_subject = np.concatenate([X1, X2])
                y_subject = np.concatenate([np.ones(len(X1)), np.zeros(len(X2))])
                
                # Perform individual subject classification
                subject_results = perform_csp_lda(
                    X_subject, 
                    y_subject, 
                    title=f"Subject {subject}: {condition1} vs {condition2}"
                )
                individual_results[subject] = subject_results
                
                # Z-score normalization per subject before combining for group analysis
                scaler = StandardScaler()
                n_trials, n_channels, n_times = X_subject.shape
                X_subject_reshaped = X_subject.reshape(n_trials, -1)
                X_subject_normalized = scaler.fit_transform(X_subject_reshaped)
                X_subject = X_subject_normalized.reshape(n_trials, n_channels, n_times)
                
                # Add to group dataset
                X_all.append(X_subject)
                y_all.append(y_subject)
                subject_ids.extend([subject] * len(X_subject))
                
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        # Store individual results
        all_individual_results[pair_name] = individual_results
        
        # Group-level analysis
        if len(X_all) > 0:
            try:
                X_group = np.concatenate(X_all)
                y_group = np.concatenate(y_all)
                subject_group = np.array(subject_ids)
                
                print(f"\nGroup analysis with {len(X_group)} total trials")
                
                # Store the first subject's info for visualization
                info = None
                for subject in subjects:
                    if (subject in eeg_data and condition1 in eeg_data[subject] and 
                        eeg_data[subject][condition1]):
                        info = eeg_data[subject][condition1][0].info
                        break
                
                # Perform group classification with subject-aware cross-validation
                group_results[pair_name] = {
                    'X': X_group,
                    'y': y_group,
                    'subjects': subject_group,
                    'info': info,
                    'channels': available_channels,
                    'condition1': condition1,
                    'condition2': condition2,
                    'times': times[start_idx:end_idx],
                    'results': perform_group_csp_lda(
                        X_group, y_group, subject_group, 
                        f"Group: {condition1} vs {condition2}"
                    )
                }
                
                # Calculate the average of individual accuracies for comparison
                individual_accuracies = [res['accuracy'] for res in individual_results.values()]
                mean_individual_acc = np.mean(individual_accuracies)
                std_individual_acc = np.std(individual_accuracies)
                group_results[pair_name]['individual_mean_acc'] = mean_individual_acc
                group_results[pair_name]['individual_std_acc'] = std_individual_acc
                
                print(f"Average individual accuracy: {mean_individual_acc:.3f} ± {std_individual_acc:.3f}")
                
            except Exception as e:
                print(f"Error in group analysis: {e}")
        else:
            print("No valid data for group analysis")
    
    return group_results, all_individual_results


# ------------------------------------------------------------------------------------- 
# FREQUENCY BAND ANALYSIS
# ------------------------------------------------------------------------------------- 

def analyze_frequency_bands(eeg_data, subjects):
    """
    Analyze classification performance across different frequency bands
    within the pre-filtered range of 6-30 Hz.
    
    Parameters:
    -----------
    eeg_data : dict
        Dictionary of EEG data organized by subject and condition
    subjects : list
        List of subject IDs
        
    Returns:
    --------
    band_results : dict
        Dictionary with classification results for each frequency band
    band_names : list
        List of frequency band names
    """
    # Define frequency bands within the 6-30 Hz range
    frequency_bands = [
        (6, 10),    # Lower alpha
        (8, 12),    # Alpha
        (12, 18),   # Lower beta
        (18, 25),   # Upper beta
        (25, 30),   # Higher beta
        (8, 16),    # Alpha + lower beta
        (16, 30),   # Higher beta
        (6, 30)     # Full band
    ]
    
    # Results storage
    band_results = {'real_vs_rest': [], 'real_vs_imagined': []}
    band_names = [f"{band[0]}-{band[1]}Hz" for band in frequency_bands]
    
    # Analyze each frequency band
    for band in frequency_bands:
        print(f"\n{'-'*50}")
        print(f"Testing frequency band {band[0]}-{band[1]} Hz")
        
        # Run classification with this frequency band
        group_results, _ = classify_condition_pairs(eeg_data, subjects, freq_band=band)
        
        # Store results
        if 'real_right_hand_vs_rest' in group_results:
            band_results['real_vs_rest'].append(
                group_results['real_right_hand_vs_rest']['individual_mean_acc'])
        else:
            band_results['real_vs_rest'].append(np.nan)
            
        if 'real_right_hand_vs_imagined_right_hand' in group_results:
            band_results['real_vs_imagined'].append(
                group_results['real_right_hand_vs_imagined_right_hand']['individual_mean_acc'])
        else:
            band_results['real_vs_imagined'].append(np.nan)
    
    # Visualize results
    plot_frequency_band_results(band_results, band_names)
    
    return band_results, band_names


def plot_frequency_band_results(band_results, band_names):
    """
    Create a bar plot comparing classification performance across frequency bands.
    
    Parameters:
    -----------
    band_results : dict
        Dictionary with classification results for each frequency band
    band_names : list
        List of frequency band names
    """
    plt.figure(figsize=(12, 7))
    
    # Set up positions
    x = np.arange(len(band_names))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, band_results['real_vs_rest'], width, 
            label='Real vs. Rest', color='#3498db')
    plt.bar(x + width/2, band_results['real_vs_imagined'], width, 
            label='Real vs. Imagined', color='#e74c3c')
    
    # Add chance level line
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    
    # Labels and formatting
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Frequency Band')
    plt.title('Classification Performance Across Frequency Bands')
    plt.xticks(x, band_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Highlight the best performing bands
    real_rest_acc = np.array(band_results['real_vs_rest'])
    real_imag_acc = np.array(band_results['real_vs_imagined'])
    
    # Handle NaN values
    if np.all(np.isnan(real_rest_acc)) or np.all(np.isnan(real_imag_acc)):
        print("Warning: All accuracy values are NaN, cannot highlight best bands")
        return
    
    # Find best bands (ignoring NaNs)
    best_real_rest = np.nanargmax(real_rest_acc)
    best_real_imag = np.nanargmax(real_imag_acc)
    
    if not np.isnan(real_rest_acc[best_real_rest]):
        plt.annotate(f"Best: {real_rest_acc[best_real_rest]:.3f}", 
                    xy=(best_real_rest - width/2, real_rest_acc[best_real_rest]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', color='#3498db', fontweight='bold')
    
    if not np.isnan(real_imag_acc[best_real_imag]):
        plt.annotate(f"Best: {real_imag_acc[best_real_imag]:.3f}", 
                    xy=(best_real_imag + width/2, real_imag_acc[best_real_imag]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', color='#e74c3c', fontweight='bold')
    
    plt.show()
    
    # Print summary of results
    print("\n" + "="*80)
    print("FREQUENCY BAND ANALYSIS SUMMARY:")
    print("="*80)
    
    if not np.isnan(real_rest_acc[best_real_rest]):
        print(f"Best frequency band for Real vs. Rest: {band_names[best_real_rest]} "
              f"({real_rest_acc[best_real_rest]:.3f})")
    
    if not np.isnan(real_imag_acc[best_real_imag]):
        print(f"Best frequency band for Real vs. Imagined: {band_names[best_real_imag]} "
              f"({real_imag_acc[best_real_imag]:.3f})")
    
    # Print neuroscientific interpretation
    print("\nNEUROSCIENTIFIC INTERPRETATION:")
    interpretation = {
        "6-10Hz": "Lower alpha band, often associated with attentional processes",
        "8-12Hz": "Alpha band, strong association with motor imagery and execution",
        "12-18Hz": "Lower beta band, linked to active motor processing",
        "18-25Hz": "Upper beta band, reflects motor preparation and inhibition",
        "25-30Hz": "Higher beta band, associated with complex cognitive processing",
        "8-16Hz": "Alpha + lower beta, encompasses classical motor rhythm (mu rhythm)",
        "16-30Hz": "Higher beta range, reflects cognitive and integrative aspects",
        "6-30Hz": "Full band, captures all relevant motor-related oscillations"
    }
    
    if not np.isnan(real_rest_acc[best_real_rest]):
        print(f"Real vs. Rest best in {band_names[best_real_rest]}: {interpretation[band_names[best_real_rest]]}")
    
    if not np.isnan(real_imag_acc[best_real_imag]):
        print(f"Real vs. Imagined best in {band_names[best_real_imag]}: {interpretation[band_names[best_real_imag]]}")


# ------------------------------------------------------------------------------------- 
# TIME WINDOW ANALYSIS
# ------------------------------------------------------------------------------------- 

def analyze_time_windows(eeg_data, subjects):
    """
    Analyze classification performance across different time windows
    within the epoch range from -1 to 3 seconds.
    
    Parameters:
    -----------
    eeg_data : dict
        Dictionary of EEG data organized by subject and condition
    subjects : list
        List of subject IDs
        
    Returns:
    --------
    window_results : dict
        Dictionary with classification results for each time window
    window_names : list
        List of time window names
    """
    # Define time windows to analyze
    time_windows = [
        (-1.0, 0.0),    # Baseline/preparation
        (0.0, 0.5),     # Early execution
        (0.5, 1.5),     # Mid execution
        (1.5, 3.0),     # Late execution/recovery
        (0.0, 3.0),     # Full execution period
        (-1.0, 3.0)     # Entire epoch
    ]
    
    # Results storage
    window_results = {'real_vs_rest': [], 'real_vs_imagined': []}
    window_names = [f"{win[0]:.1f}-{win[1]:.1f}s" for win in time_windows]
    
    # Analyze each time window
    for time_window in time_windows:
        print(f"\n{'-'*50}")
        print(f"Testing time window {time_window[0]:.1f}-{time_window[1]:.1f} seconds")
        
        # Run classification with this time window
        group_results, _ = classify_condition_pairs_with_time_window(
            eeg_data, subjects, time_window=time_window)
        
        # Store results (handling possible missing keys)
        if 'real_right_hand_vs_rest' in group_results:
            window_results['real_vs_rest'].append(
                group_results['real_right_hand_vs_rest']['individual_mean_acc'])
        else:
            window_results['real_vs_rest'].append(np.nan)
            
        if 'real_right_hand_vs_imagined_right_hand' in group_results:
            window_results['real_vs_imagined'].append(
                group_results['real_right_hand_vs_imagined_right_hand']['individual_mean_acc'])
        else:
            window_results['real_vs_imagined'].append(np.nan)
    
    # Visualize results
    plot_time_window_results(window_results, window_names)
    
    return window_results, window_names


def classify_condition_pairs_with_time_window(eeg_data, subjects, time_window=None, freq_band=(8, 30)):
    """
    Perform CSP+LDA classification on multiple condition pairs with a specific time window.
    
    Parameters:
    -----------
    eeg_data : dict
        Dictionary of EEG data organized by subject and condition
    subjects : list
        List of subject IDs
    time_window : tuple
        Specific time window to extract (start_time, end_time) in seconds
    freq_band : tuple
        Frequency band for filtering
        
    Returns:
    --------
    group_results : dict
        Dictionary containing group-level results for different condition pairs
    all_individual_results : dict
        Dictionary containing all individual subject results
    """
    # Define condition pairs to analyze
    condition_pairs = [
        ('real_right_hand', 'rest'),
        ('real_right_hand', 'imagined_right_hand')
    ]
    
    # Results dictionary for group analysis
    group_results = {}
    all_individual_results = {}
    
    # Process each condition pair
    for condition_pair in condition_pairs:
        condition1, condition2 = condition_pair
        pair_name = f"{condition1}_vs_{condition2}"
        print(f"\n{'='*80}\nAnalyzing {condition1} vs {condition2} (Time window: {time_window})\n{'='*80}")
        
        # Process all subjects for this condition pair
        X_all = []
        y_all = []
        subject_ids = []
        individual_results = {}
        
        for subject in subjects:
            print(f"\nProcessing subject: {subject}")
            
            try:
                # Check for required conditions
                if not (condition1 in eeg_data[subject] and condition2 in eeg_data[subject]):
                    print(f"Skipping subject {subject}: missing condition data")
                    continue
                
                # Skip if no epochs available
                if not eeg_data[subject][condition1] or not eeg_data[subject][condition2]:
                    print(f"Skipping subject {subject}: empty epoch list")
                    continue
                    
                # Concatenate epochs for this subject
                epochs1 = mne.concatenate_epochs(eeg_data[subject][condition1])
                epochs2 = mne.concatenate_epochs(eeg_data[subject][condition2])
                
                # Apply bandpass filter
                fmin, fmax = freq_band
                epochs1 = epochs1.copy().filter(fmin, fmax)
                epochs2 = epochs2.copy().filter(fmin, fmax)
                
                # Select motor channels
                motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'FC1', 'FC2', 'CP1', 'CP2']
                available_channels = [ch for ch in motor_channels if ch in epochs1.ch_names]
                
                if not available_channels:
                    print(f"Skipping subject {subject}: no motor channels available")
                    continue
                
                # Pick channels
                epochs1_motor = epochs1.copy().pick_channels(available_channels)
                epochs2_motor = epochs2.copy().pick_channels(available_channels)
                
                # Get time indices based on the specified time window
                times = epochs1_motor.times
                
                if time_window is None:
                    # Default: use execution phase
                    start_idx = np.where(times >= 0)[0][0]
                    end_idx = np.where(times >= 2.0)[0][0] if any(times >= 2.0) else -1
                else:
                    # Use specified time window
                    start_time, end_time = time_window
                    
                    # Safely find indices that fall within epoch time range
                    start_time_idx = np.where(times >= start_time)[0]
                    end_time_idx = np.where(times >= end_time)[0]
                    
                    if len(start_time_idx) == 0:
                        print(f"Warning: Start time {start_time}s is outside epoch range. Using first index.")
                        start_idx = 0
                    else:
                        start_idx = start_time_idx[0]
                        
                    if len(end_time_idx) == 0:
                        print(f"Warning: End time {end_time}s is outside epoch range. Using last index.")
                        end_idx = len(times) - 1
                    else:
                        end_idx = end_time_idx[0]
                
                # Extract data with specified time window
                X1 = epochs1_motor.get_data()[:, :, start_idx:end_idx]
                X2 = epochs2_motor.get_data()[:, :, start_idx:end_idx]
                
                # Balance classes through undersampling
                if len(X2) > len(X1):
                    indices = np.random.choice(len(X2), len(X1), replace=False)
                    X2 = X2[indices]
                    print(f"Subsampled {condition2} to match {condition1}: {len(X1)} trials")
                
                # Create combined dataset for this subject
                X_subject = np.concatenate([X1, X2])
                y_subject = np.concatenate([np.ones(len(X1)), np.zeros(len(X2))])
                
                # Perform individual subject classification
                subject_results = perform_csp_lda(
                    X_subject, 
                    y_subject, 
                    title=f"Subject {subject}: {condition1} vs {condition2}"
                )
                individual_results[subject] = subject_results
                
                # Z-score normalization per subject before combining
                scaler = StandardScaler()
                n_trials, n_channels, n_times = X_subject.shape
                X_subject_reshaped = X_subject.reshape(n_trials, -1)
                X_subject_normalized = scaler.fit_transform(X_subject_reshaped)
                X_subject = X_subject_normalized.reshape(n_trials, n_channels, n_times)
                
                # Add to group dataset
                X_all.append(X_subject)
                y_all.append(y_subject)
                subject_ids.extend([subject] * len(X_subject))
                
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        # Store individual results
        all_individual_results[pair_name] = individual_results
        
        # Group-level analysis
        if len(X_all) > 0:
            try:
                X_group = np.concatenate(X_all)
                y_group = np.concatenate(y_all)
                subject_group = np.array(subject_ids)
                
                print(f"\nGroup analysis with {len(X_group)} total trials")
                
                # Use the first subject's info for visualization
                info = None
                for subject in subjects:
                    if (subject in eeg_data and condition1 in eeg_data[subject] and 
                        eeg_data[subject][condition1]):
                        info = eeg_data[subject][condition1][0].info
                        break
                
                # Perform group classification with subject-aware cross-validation
                group_results[pair_name] = {
                    'X': X_group,
                    'y': y_group,
                    'subjects': subject_group,
                    'info': info,
                    'channels': available_channels,
                    'condition1': condition1,
                    'condition2': condition2,
                    'times': times[start_idx:end_idx],
                    'results': perform_group_csp_lda(
                        X_group, y_group, subject_group, 
                        f"Group: {condition1} vs {condition2}"
                    )
                }
                
                # Calculate the average of individual accuracies for comparison
                individual_accuracies = [res['accuracy'] for res in individual_results.values()]
                mean_individual_acc = np.mean(individual_accuracies)
                std_individual_acc = np.std(individual_accuracies)
                group_results[pair_name]['individual_mean_acc'] = mean_individual_acc
                group_results[pair_name]['individual_std_acc'] = std_individual_acc
                
                print(f"Average individual accuracy: {mean_individual_acc:.3f} ± {std_individual_acc:.3f}")
                
            except Exception as e:
                print(f"Error in group analysis: {e}")
        else:
            print("No valid data for group analysis")
    
    return group_results, all_individual_results


def plot_time_window_results(window_results, window_names):
    """
    Create a line plot showing how classification performance changes across time windows.
    
    Parameters:
    -----------
    window_results : dict
        Dictionary with classification results for each time window
    window_names : list
        List of time window names
    """
    plt.figure(figsize=(12, 7))
    
    # Set up positions
    x = np.arange(len(window_names))
    
    # Convert to numpy arrays to handle NaNs properly
    real_rest_acc = np.array(window_results['real_vs_rest'])
    real_imag_acc = np.array(window_results['real_vs_imagined'])
    
    # Plot lines with markers
    plt.plot(x, real_rest_acc, 'o-', 
             linewidth=2, markersize=8, label='Real vs. Rest', color='#3498db')
    plt.plot(x, real_imag_acc, 's-', 
             linewidth=2, markersize=8, label='Real vs. Imagined', color='#e74c3c')
    
    # Add chance level line
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    
    # Add vertical line at movement onset (t=0)
    # Find index of time window with 0.0 as start time
    onset_idx = [i for i, name in enumerate(window_names) if name.startswith('0.0-')]
    if onset_idx:
        plt.axvline(x=onset_idx[0], color='green', linestyle=':', 
                    label='Movement Onset', alpha=0.7)
    
    # Labels and formatting
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Time Window (seconds)')
    plt.title('Classification Performance Across Time Windows')
    plt.xticks(x, window_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Highlight the best performing time windows if they exist and are not NaN
    if not np.all(np.isnan(real_rest_acc)):
        best_real_rest = np.nanargmax(real_rest_acc)
        plt.annotate(f"Best: {real_rest_acc[best_real_rest]:.3f}", 
                    xy=(best_real_rest, real_rest_acc[best_real_rest]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', color='#3498db', fontweight='bold')
    
    if not np.all(np.isnan(real_imag_acc)):
        best_real_imag = np.nanargmax(real_imag_acc)
        plt.annotate(f"Best: {real_imag_acc[best_real_imag]:.3f}", 
                    xy=(best_real_imag, real_imag_acc[best_real_imag]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', color='#e74c3c', fontweight='bold')
    
    plt.show()
    
    # Print summary of results
    print("\n" + "="*80)
    print("TIME WINDOW ANALYSIS SUMMARY:")
    print("="*80)
    
    if not np.all(np.isnan(real_rest_acc)):
        best_real_rest = np.nanargmax(real_rest_acc)
        print(f"Best time window for Real vs. Rest: {window_names[best_real_rest]} "
              f"({real_rest_acc[best_real_rest]:.3f})")
    
    if not np.all(np.isnan(real_imag_acc)):
        best_real_imag = np.nanargmax(real_imag_acc)
        print(f"Best time window for Real vs. Imagined: {window_names[best_real_imag]} "
              f"({real_imag_acc[best_real_imag]:.3f})")
    
    # Print neuroscientific interpretation
    print("\nNEUROSCIENTIFIC INTERPRETATION:")
    interpretation = {
        "-1.0-0.0s": "Preparation phase, anticipatory activity",
        "0.0-0.5s": "Early movement execution, movement initiation",
        "0.5-1.5s": "Sustained movement, active motor control",
        "1.5-3.0s": "Late phase, potentially includes termination and recovery",
        "0.0-3.0s": "Entire execution phase",
        "-1.0-3.0s": "Full trial including preparation and execution"
    }
    
    if not np.all(np.isnan(real_rest_acc)):
        best_real_rest = np.nanargmax(real_rest_acc)
        print(f"Real vs. Rest best in {window_names[best_real_rest]}: {interpretation[window_names[best_real_rest]]}")
    
    if not np.all(np.isnan(real_imag_acc)):
        best_real_imag = np.nanargmax(real_imag_acc)
        print(f"Real vs. Imagined best in {window_names[best_real_imag]}: {interpretation[window_names[best_real_imag]]}")




# ------------------------------------------------------------------------------------- 
# PATTERN SIMILARITY AND DIMENSIONALITY REDUCTION
# ------------------------------------------------------------------------------------- 


def compute_rsa_and_plot_mds(epochs_dict, picks=None, metric="correlation"):
    """
    Compute Representational Similarity Analysis (RSA) and visualize using MDS.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary with condition names as keys and mne.Epochs as values
    picks : list or None
        Channel picks. If None, all channels are used
    metric : str
        Distance metric to use (default = 'correlation')
        
    Returns:
    --------
    dist_matrix : ndarray
        Pairwise distance matrix between conditions
    coords : ndarray
        2D coordinates from MDS projection
    """
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    
    condition_names = list(epochs_dict.keys())
    
    # Validate input
    if len(condition_names) < 2:
        raise ValueError("At least two conditions required for comparison")
    
    # Check that epochs have the same channels if picks is None
    if picks is None:
        ch_names_sets = [set(epochs.ch_names) for epochs in epochs_dict.values()]
        if len(set.intersection(*ch_names_sets)) == 0:
            raise ValueError("No common channels found across conditions")
        
    avg_patterns = []

    # Step 1: Compute mean spatial pattern for each condition
    for cond in condition_names:
        if cond not in epochs_dict:
            raise ValueError(f"Condition '{cond}' not found in epochs_dict")
            
        data = epochs_dict[cond].get_data(picks=picks)  # shape: (n_epochs, n_channels, n_times)
        
        if data.size == 0:
            raise ValueError(f"No data found for condition '{cond}' with picks {picks}")
            
        avg_pattern = np.mean(data, axis=(0, 2))  # average over epochs and time -> shape: (n_channels,)
        avg_patterns.append(avg_pattern)

    avg_patterns = np.array(avg_patterns)

    # Step 2: Compute pairwise dissimilarities
    dist_matrix = pairwise_distances(avg_patterns, metric=metric)

    # Step 3: MDS to project into 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)

    # Step 4: Plot
    plt.figure(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(condition_names)))
    
    for i, cond in enumerate(condition_names):
        plt.scatter(coords[i, 0], coords[i, 1], label=cond.upper(), 
                   s=100, color=colors[i])
        plt.text(coords[i, 0], coords[i, 1], cond.upper(), 
                fontsize=12, ha='center', va='center')

    plt.title("RSA (MDS Projection of Condition Similarities)")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print distance matrix
    print("\nCondition Similarity (Distance Matrix):")
    similarity_df = pd.DataFrame(dist_matrix, 
                               index=condition_names, 
                               columns=condition_names)
    print(similarity_df)

    return dist_matrix, coords


def plot_tsne_all_trials(epochs_dict, picks=None, perplexity=30, n_components=2, max_samples=500):
    """
    Flatten and reduce all trials across conditions using t-SNE.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary with condition names as keys and mne.Epochs as values
    picks : list or None
        Channel picks. If None, all channels are used
    perplexity : int
        Perplexity parameter for t-SNE (default=30)
    n_components : int
        Number of dimensions for t-SNE output (default=2)
    max_samples : int or None
        Maximum number of trials to include (downsampling for computational efficiency)
        If None, all trials are used
        
    Returns:
    --------
    X_tsne : ndarray
        Reduced dimensionality representation from t-SNE
    y : list
        Condition labels for each point in X_tsne
    """
    from sklearn.manifold import TSNE
    
    X = []
    y = []
    n_samples_per_condition = {}

    for label, epochs in epochs_dict.items():
        data = epochs.get_data(picks=picks)  # shape: (n_epochs, n_channels, n_times)
        
        # Skip empty data
        if data.shape[0] == 0:
            print(f"Warning: No data for condition '{label}', skipping")
            continue
            
        n_samples_per_condition[label] = data.shape[0]
        
        # Flatten the data
        flat = data.reshape(data.shape[0], -1)  # flatten each epoch to (n_epochs, n_features)
        X.append(flat)
        y.extend([label] * len(flat))

    if not X:
        raise ValueError("No valid data found in any condition")
        
    X = np.vstack(X)
    total_samples = X.shape[0]
    
    # Downsample if needed
    if max_samples is not None and total_samples > max_samples:
        print(f"Downsampling from {total_samples} to {max_samples} trials...")
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X = X[indices]
        y = [y[i] for i in indices]
    
    # Run t-SNE
    print(f"Running t-SNE on {X.shape[0]} trials with {X.shape[1]} features...")
    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, X.shape[0]-1), 
                random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    # Plot for 2D case
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            idx = [j for j, val in enumerate(y) if val == label]
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                       label=f"{label} (n={n_samples_per_condition.get(label, len(idx))})", 
                       alpha=0.7, color=colors[i])
                       
        plt.title("Trial-wise t-SNE Projection")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    # Print statistical summary
    print("\nSample distribution:")
    for label in np.unique(y):
        count = y.count(label)
        print(f"  {label}: {count} trials ({count/len(y)*100:.1f}%)")

    return X_tsne, y


def compute_topographic_distances(epochs_dict, picks=None):
    """
    Compute pairwise distances between condition topographies.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary with condition names as keys and mne.Epochs as values
    picks : list or None
        Channel picks. If None, all channels are used
        
    Returns:
    --------
    distances : dict
        Dictionary with pairwise distance metrics
    topographies : dict
        Dictionary with average spatial patterns for each condition
    """
    from scipy.spatial.distance import cosine, euclidean, correlation
    
    # Extract topographic patterns
    topographies = {}
    for cond, epochs in epochs_dict.items():
        data = epochs.get_data(picks=picks)
        
        if data.size == 0:
            print(f"Warning: No data for condition '{cond}', skipping")
            continue
            
        # Average over epochs and time points
        topographies[cond] = np.mean(data, axis=(0, 2))
    
    if len(topographies) < 2:
        raise ValueError("At least two valid conditions required for distance computation")
    
    # Compute pairwise distances
    labels = list(topographies.keys())
    distances = {}
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            cond1, cond2 = labels[i], labels[j]
            v1, v2 = topographies[cond1], topographies[cond2]
            
            # Compute multiple distance metrics
            distances[f"{cond1} vs {cond2}"] = {
                "cosine": cosine(v1, v2),
                "euclidean": euclidean(v1, v2),
                "correlation": correlation(v1, v2)
            }
    
    # Create results table
    results = []
    for pair, metrics in distances.items():
        results.append({
            "Comparison": pair,
            "Cosine Distance": metrics["cosine"],
            "Euclidean Distance": metrics["euclidean"],
            "Correlation Distance": metrics["correlation"]
        })
    
    # Display results
    print("\nTopographic Distance Metrics:")
    distance_df = pd.DataFrame(results)
    print(distance_df)
    
    # Visualize topographic patterns
    n_conds = len(topographies)
    fig, axes = plt.subplots(1, n_conds, figsize=(4*n_conds, 4))
    
    if n_conds == 1:
        axes = [axes]  # Handle case with only one condition
        
    for i, (cond, topo) in enumerate(topographies.items()):
        # Get channel info if available
        info = epochs_dict[cond].info if hasattr(epochs_dict[cond], 'info') else None
        
        if info and picks is None:
            # Plot as topographic map if we have channel positions
            try:
                mne.viz.plot_topomap(topo, info, axes=axes[i], show=False)
                axes[i].set_title(f"{cond}")
            except Exception as e:
                # Fall back to bar plot if topographic plotting fails
                print(f"Could not create topomap for {cond}: {e}")
                axes[i].bar(range(len(topo)), topo)
                axes[i].set_title(f"{cond}")
        else:
            # Simple bar plot of values
            axes[i].bar(range(len(topo)), topo)
            axes[i].set_title(f"{cond}")
    
    plt.tight_layout()
    plt.show()
            
    return distances, topographies


def analyze_spatial_patterns(epochs_dict, picks=None, run_tsne=True):
    """
    Perform comprehensive spatial pattern analysis including RSA, 
    topographic distances, and optionally t-SNE.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary with condition names as keys and mne.Epochs as values
    picks : list or None
        Channel picks. If None, all channels are used
    run_tsne : bool
        Whether to run t-SNE analysis (computationally intensive)
        
    Returns:
    --------
    results : dict
        Dictionary with results from all analyses
    """
    results = {}
    
    # 1. RSA and MDS visualization
    print("\n" + "="*80)
    print("REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)")
    print("="*80)
    try:
        dist_matrix, coords = compute_rsa_and_plot_mds(epochs_dict, picks)
        results['rsa'] = {
            'distance_matrix': dist_matrix,
            'mds_coordinates': coords
        }
    except Exception as e:
        print(f"Error in RSA analysis: {e}")
        results['rsa'] = None
    
    # 2. Topographic distance analysis
    print("\n" + "="*80)
    print("TOPOGRAPHIC DISTANCE ANALYSIS")
    print("="*80)
    try:
        distances, topographies = compute_topographic_distances(epochs_dict, picks)
        results['topographic'] = {
            'distances': distances,
            'patterns': topographies
        }
    except Exception as e:
        print(f"Error in topographic distance analysis: {e}")
        results['topographic'] = None
    
    # 3. Optional t-SNE analysis
    if run_tsne:
        print("\n" + "="*80)
        print("t-SNE TRIAL PROJECTION")
        print("="*80)
        try:
            X_tsne, labels = plot_tsne_all_trials(epochs_dict, picks)
            results['tsne'] = {
                'coordinates': X_tsne,
                'labels': labels
            }
        except Exception as e:
            print(f"Error in t-SNE analysis: {e}")
            results['tsne'] = None
    
    return results