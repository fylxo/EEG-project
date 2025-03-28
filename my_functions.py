"""
Functions for analyzing EEG data related to motor movement and imagery.
Includes preprocessing, connectivity analysis, and classification functions.
"""

# Core libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Step 2: Apply Bonferroni correction (only to valid p-values)
    valid_p_real_rest = [p for p in p_values_real_rest if not np.isnan(p)]
    valid_p_real_imagined = [p for p in p_values_real_imagined if not np.isnan(p)]
    
    if valid_p_real_rest:
        _, p_real_rest_corr, _, _ = multipletests(valid_p_real_rest, alpha=alpha, method='bonferroni')
        
        # Map corrected p-values back to results
        p_idx = 0
        for i, res in enumerate(results):
            if not np.isnan(res["p Real vs Rest"]):
                res["p Real vs Rest (Bonf)"] = p_real_rest_corr[p_idx]
                p_idx += 1
            else:
                res["p Real vs Rest (Bonf)"] = np.nan
    
    if valid_p_real_imagined:
        _, p_real_imagined_corr, _, _ = multipletests(valid_p_real_imagined, alpha=alpha, method='bonferroni')
        
        # Map corrected p-values back to results
        p_idx = 0
        for i, res in enumerate(results):
            if not np.isnan(res["p Real vs Imagined"]):
                res["p Real vs Imagined (Bonf)"] = p_real_imagined_corr[p_idx]
                p_idx += 1
            else:
                res["p Real vs Imagined (Bonf)"] = np.nan

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