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

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from mne.time_frequency import tfr_morlet


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


# ------------------------------------------------------------------------------------- PREPROCESSING PIPELINE -------------------------------------------------------------------------------------

def process_eeg_automatic(subjects, data_path, apply_ica=False, bandpass=(6, 30), n_ica_components=25, tmin=-1.0, tmax=3.0):
    """
    Loads EEG data, applies preprocessing, ICA artifact removal, extracts epochs, and stores them by condition.
    """
    mne.set_log_level("ERROR")  # Reduce MNE verbosity
    data_dict = {}

    # Define selected runs
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
                    raw = mne.io.read_raw_edf(edf_file, preload=True)

                    # Standardize channel names and set montage
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)

                    # Set EEG reference
                    raw.set_eeg_reference("average", projection=True)

                    # Apply band-pass filter
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # Apply ICA if enabled
                    if apply_ica:
                        print(f"🔍 Running ICA for {subject} - {run}...")
                        ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=42, method="fastica", max_iter=500)
                        ica.fit(raw)

                        # **1. Detect EOG Artifacts Using Frontal Channels**
                        print(f"  👁️ Detecting eye artifacts...")
                        eog_indices = []
                        frontal_channels = [ch for ch in raw.ch_names if ch.lower().startswith(('fp', 'f')) 
                                            and not ch.lower().startswith(('fc', 'ft'))][:2]  # Select up to 2 frontal channels

                        if frontal_channels:
                            eog_indices, _ = ica.find_bads_eog(raw, ch_name=frontal_channels)
                            print(f"    Using channels: {frontal_channels}")

                        eog_indices = eog_indices[:2] if len(eog_indices) > 2 else eog_indices
                        print(f"  👁️ EOG components: {eog_indices}")

                        # **2. Detect ECG Artifacts Using Kurtosis**
                        print(f"💓 Detecting cardiac artifacts using kurtosis...")
                        ica_sources = ica.get_sources(raw).get_data()
                        kurtosis_values = scipy.stats.kurtosis(ica_sources, axis=1)

                        # Z-score transformation for kurtosis-based detection
                        kurt_z = (kurtosis_values - np.median(kurtosis_values)) / (
                            np.median(np.abs(kurtosis_values - np.median(kurtosis_values))) + 1e-6
                        )
                        ecg_indices = np.where(kurt_z > 2.5)[0].tolist()[:2]  # Select top 2 cardiac-like components
                        print(f"💓 ECG artifact components (estimated): {ecg_indices}")

                        # **3. Detect Muscle Artifacts**
                        print(f"💪 Detecting muscle artifacts...")
                        muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.5, l_freq=7, h_freq=45)
                        print(f"💪 Muscle artifact components: {muscle_indices}")

                        # **4. Combine and exclude bad components**
                        all_artifact_indices = list(set(eog_indices + ecg_indices + muscle_indices))
                        ica.exclude = all_artifact_indices
                        print(f"❌ Removing {len(ica.exclude)} ICA components for {subject} - {run}: {ica.exclude}")

                        # Apply ICA
                        raw_clean = raw.copy()
                        ica.apply(raw_clean)
                    else:
                        raw_clean = raw  # Use raw data without ICA

                    # Get events
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

    print("\n✅ Processing complete for the subject.")
    return data_dict


def process_eeg_manual(subjects, data_path, bandpass=(6, 30), n_ica_components=25, tmin=-1.0, tmax=2.0, manual_review=True):
    """
    Loads EEG data, applies preprocessing, ICA artifact removal, extracts epochs, and stores them by condition.
    """
    mne.set_log_level("ERROR")  # Reduce MNE verbosity
    data_dict = {}

    # Define selected runs
    selected_runs = {
        "real_right_hand": [3, 7, 11],
        "imagined_right_hand": [4, 8, 12]
    }

    for subject in subjects:
        print(f"\n🔄 Processing {subject}...")
        data_dict[subject] = {"real_right_hand": [], "imagined_right_hand": []}

        for condition, run_numbers in selected_runs.items():
            for run_number in run_numbers:
                run = f"R{str(run_number).zfill(2)}"
                edf_file = os.path.join(data_path, subject, f"{subject}{run}.edf")

                try:
                    raw = mne.io.read_raw_edf(edf_file, preload=True)

                    # Standardize channel names and set montage
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)

                    # Set EEG reference
                    raw.set_eeg_reference("average", projection=True)

                    # Apply band-pass filter
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # Apply ICA if enabled
                    if apply_ica:
                        print(f"🔍 Running ICA for {subject} - {run}...")
                        ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=42, method="fastica", max_iter=500)
                        ica.fit(raw)

                        # **1. Detect EOG Artifacts Using Frontal Channels**
                        print(f"  👁️ Detecting eye artifacts...")
                        eog_indices = []
                        frontal_channels = [ch for ch in raw.ch_names if ch.lower().startswith(('fp', 'f')) 
                                            and not ch.lower().startswith(('fc', 'ft'))][:2]  # Select up to 2 frontal channels

                        if frontal_channels:
                            eog_indices, _ = ica.find_bads_eog(raw, ch_name=frontal_channels)
                            print(f"    Using channels: {frontal_channels}")

                        eog_indices = eog_indices[:2] if len(eog_indices) > 2 else eog_indices
                        print(f"  👁️ EOG components: {eog_indices}")

                        # **2. Detect ECG Artifacts Using Kurtosis**
                        print(f"💓 Detecting cardiac artifacts using kurtosis...")
                        ica_sources = ica.get_sources(raw).get_data()
                        kurtosis_values = scipy.stats.kurtosis(ica_sources, axis=1)

                        # Z-score transformation for kurtosis-based detection
                        kurt_z = (kurtosis_values - np.median(kurtosis_values)) / (
                            np.median(np.abs(kurtosis_values - np.median(kurtosis_values))) + 1e-6
                        )
                        ecg_indices = np.where(kurt_z > 2.5)[0].tolist()[:2]  # Select top 2 cardiac-like components
                        print(f"💓 ECG artifact components (estimated): {ecg_indices}")

                        # **3. Detect Muscle Artifacts**
                        print(f"💪 Detecting muscle artifacts...")
                        muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.5, l_freq=7, h_freq=45)
                        print(f"💪 Muscle artifact components: {muscle_indices}")

                        # **4. Combine automatic exclusions first**
                        all_artifact_indices = list(set(eog_indices + ecg_indices + muscle_indices))
                        ica.exclude = all_artifact_indices
                        print(f"⚠️ Automated exclusions: {ica.exclude}")

                        # **5. Manual review option**
                        if manual_review:
                            print(f"🔎 Manual inspection time! Review ICA components for {subject} - {run}")
                            ica.plot_components(inst=raw)
                            plt.show()

                            # Optional: Look at time series + spectra of components marked by automation
                            if len(ica.exclude) > 0:
                                ica.plot_properties(raw, picks=ica.exclude)

                            # After manual tweaking, the user can close the plot window and `ica.exclude` will retain final selections
                            print(f"✅ Manual exclusions now: {ica.exclude}")

                        # **6. Apply ICA after review**
                        raw_clean = raw.copy()
                        ica.apply(raw_clean)
                    else:
                        raw_clean = raw  # Use raw data without ICA

                    # Get events
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

    print("\n✅ Processing complete for the subject.")
    return data_dict


def process_eeg_debug(subjects, data_path, bandpass=(6, 30), n_ica_components=25, tmin=-1.0, tmax=2.0):
    """
    Loads EEG data, applies ICA artifact removal, and generates plots for debugging:
    - Raw data before ICA
    - ICA components and exclusions
    - Raw data after ICA
    """
    mne.set_log_level("ERROR")
    data_dict = {}

    # Define selected runs
    selected_runs = {
        "real_right_hand": [3, 7, 11],
        "imagined_right_hand": [4, 8, 12]
    }

    for subject in subjects:
        print(f"\n🔄 DEBUG Processing {subject}...")
        data_dict[subject] = {"real_right_hand": [], "imagined_right_hand": []}

        for condition, run_numbers in selected_runs.items():
            for run_number in run_numbers:
                run = f"R{str(run_number).zfill(2)}"
                edf_file = os.path.join(data_path, subject, f"{subject}{run}.edf")

                try:
                    raw = mne.io.read_raw_edf(edf_file, preload=True)
                    mne.datasets.eegbci.standardize(raw)
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw.set_montage(montage)
                    raw.set_eeg_reference("average", projection=True)
                    raw.filter(bandpass[0], bandpass[1], fir_design="firwin")

                    # 🔍 Plot raw EEG BEFORE ICA
                    print(f"📊 Plotting RAW EEG before ICA - {subject} - {run}")
                    plot_raw(raw, title=f"{subject} - {run} | Raw EEG BEFORE ICA")

                    # ---------- ICA ----------
                    print(f"🔍 Running ICA for {subject} - {run}...")
                    ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=42, method="fastica", max_iter=500)
                    ica.fit(raw)

                    # -------- Artifact Detection --------
                    eog_indices, ecg_indices, muscle_indices = [], [], []

                    # EOG
                    frontal_channels = [ch for ch in raw.ch_names if ch.lower().startswith(('fp', 'f')) and not ch.lower().startswith(('fc', 'ft'))][:2]
                    if frontal_channels:
                        eog_indices, _ = ica.find_bads_eog(raw, ch_name=frontal_channels)
                    
                    # ECG (kurtosis)
                    ica_sources = ica.get_sources(raw).get_data()
                    kurtosis_values = scipy.stats.kurtosis(ica_sources, axis=1)
                    kurt_z = (kurtosis_values - np.median(kurtosis_values)) / (np.median(np.abs(kurtosis_values - np.median(kurtosis_values))) + 1e-6)
                    ecg_indices = np.where(kurt_z > 2.5)[0].tolist()[:2]

                    # Muscle
                    muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.5, l_freq=7, h_freq=45)

                    all_artifact_indices = list(set(eog_indices + ecg_indices + muscle_indices))
                    ica.exclude = all_artifact_indices
                    print(f"⚠️ Automated exclusions: {ica.exclude}")

                    # -------- Plot ICA Components --------
                    print(f"🧠 Plotting ICA components for {subject} - {run}")
                    ica.plot_components(inst=raw)
                    plt.show()

                    # -------- Plot properties of excluded components --------
                    if len(ica.exclude) > 0:
                        print(f"📈 Plotting properties of excluded components")
                        ica.plot_properties(raw, picks=ica.exclude)
                        plt.show()

                    # ---------- Apply ICA ----------
                    raw_clean = raw.copy()
                    ica.apply(raw_clean)

                    # 🔍 Plot raw EEG AFTER ICA
                    print(f"📊 Plotting RAW EEG after ICA - {subject} - {run}")
                    plot_raw(raw_clean, title=f"{subject} - {run} | Raw EEG AFTER ICA")

                    # ---------- Epoching ----------
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

    print("\n✅ DEBUG processing complete for the subject.")
    return data_dict


def process_eeg(subjects, data_path, apply_ica=False, mode="automatic", **kwargs):
    """
    Wrapper for EEG processing pipeline with different modes.
    Modes:
        - "automatic": automatic ICA rejection
        - "manual": interactive ICA review
        - "debug": plots before/after ICA & diagnostics
    """
    if mode == "automatic":
        return process_eeg_automatic(subjects, data_path, apply_ica=apply_ica, **kwargs)
    elif mode == "manual":
        return process_eeg_manual(subjects, data_path, **kwargs)
    elif mode == "debug":
        return process_eeg_debug(subjects, data_path, **kwargs)
    else:
        raise ValueError("❌ Invalid mode! Choose from 'automatic', 'manual', or 'debug'.")


def compute_erd_ers(epochs, baseline=(-1, 0), fmin=6, fmax=30):
    """
    Compute Event-Related Desynchronization (ERD) & Synchronization (ERS).
    Uses time-frequency analysis instead of static PSD.
    """
    freqs = np.linspace(fmin, fmax, num=10)  # Define frequencies of interest
    n_cycles = freqs / 2  # Adjust cycles per frequency

    # Ensure `average=False` to keep the epoch dimension
    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, decim=3, average=False
    )  # Time-frequency power computation

    # Ensure baseline_mask correctly matches power.times
    baseline_mask = (power.times >= baseline[0]) & (power.times <= baseline[1])

    # Compute baseline average correctly (mean across time)
    baseline_power = np.mean(power.data[:, :, :, baseline_mask], axis=-1)  # Compute baseline avg

    # Compute ERD/ERS as % change from baseline
    erd_ers = 100 * (power.data - baseline_power[:, :, :, np.newaxis]) / baseline_power[:, :, :, np.newaxis]

    
    return erd_ers, power.times, power.freqs




# ------------------------------------------------------------------------------------- PLV AND COHERENCE FUNCTIONS -------------------------------------------------------------------------------------

def compute_plv_matrix(epochs):
    """
    Compute full PLV matrix across all channels in an Epochs object.
    Returns a n_channels x n_channels matrix.
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
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

    Returns:
    - plv time series
    - mean PLV value
    - max PLV value
    """
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    data = epochs.get_data(picks=[ch1, ch2])  # (n_epochs, 2, n_times)

    if data.shape[0] == 0:
        return np.nan, np.nan, np.nan

    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    phase_diff = np.exp(1j * (phase_data[:, 0, :] - phase_data[:, 1, :]))
    plv = np.abs(np.mean(phase_diff, axis=0))  # Time-resolved PLV curve

    plv_mean = np.mean(plv)
    plv_max = np.max(plv)

    return plv, plv_mean, plv_max


def compute_coherence(epochs, ch1="C3", ch2="C4", fmin=8, fmax=30, nperseg=256):
    """
    Compute coherence between two EEG channels using Welch's method.

    Parameters:
    - epochs: MNE Epochs object
    - ch1, ch2: Channel names (e.g., "C3", "C4")
    - fmin, fmax: Frequency range for coherence computation
    - nperseg: Length of each segment for Welch’s method

    Returns:
    - freqs: Frequency vector
    - coherence_values: Coherence spectrum in the selected band
    - coherence_mean: Mean coherence in the frequency range
    - coherence_max: Max coherence in the frequency range
    """
    # Ensure the channels exist
    if ch1 not in epochs.ch_names or ch2 not in epochs.ch_names:
        raise ValueError(f"Channels {ch1} and {ch2} not found in EEG data.")

    sfreq = epochs.info['sfreq']  # Sampling frequency

    # Get indices of the selected channels
    ch1_idx = epochs.ch_names.index(ch1)
    ch2_idx = epochs.ch_names.index(ch2)

    # Extract channel data and compute the average signal over trials
    data = epochs.get_data(picks=[ch1_idx, ch2_idx])  # Shape: (n_epochs, 2, n_times)
    data_avg = np.mean(data, axis=0)  # Average across trials, shape (2, n_times)

    # Compute coherence using Welch’s method
    freqs, coh_values = coherence(data_avg[0, :], data_avg[1, :], fs=sfreq, nperseg=nperseg)

    # Select coherence values in the desired frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    selected_frequencies = freqs[freq_mask]
    selected_coherence = coh_values[freq_mask]

    # Compute mean and max coherence
    coherence_mean = np.mean(selected_coherence)
    coherence_max = np.max(selected_coherence)

    return selected_frequencies, selected_coherence, coherence_mean, coherence_max


def analyze_pairwise_plv_coherence(subjects, eeg_data, conditions, channel_pairs, fmin=8, fmax=30):
    """
    Computes PLV and Coherence for multiple subject-channel pairs.

    Parameters:
    - subjects: list of subject IDs
    - eeg_data: dictionary with subject-level preprocessed data
    - conditions: dict like {"real": "real_right_hand", "imagined": "imagined_right_hand"}
    - channel_pairs: list of tuples with channel pairs (e.g., [("C3", "C4")])
    - fmin, fmax: frequency band

    Returns:
    - DataFrame with PLV & coherence values for all pairs, subjects & conditions
    """
    results = []

    for subject in subjects:
        print(f"🧠 Subject: {subject}")

        for cond_key, cond_value in conditions.items():
            print(f"  ➡️ Condition: {cond_key}")

            # Merge runs
            epochs = mne.concatenate_epochs(eeg_data[subject][cond_value])

            for ch1, ch2 in channel_pairs:
                # ----- PLV -----
                plv, plv_mean, plv_max = compute_plv_pairwise(epochs, ch1=ch1, ch2=ch2)

                # ----- Coherence -----
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

    df = pd.DataFrame(results)
    return df


def paired_ttest_plv(df, metric="PLV Mean", alpha=0.05):
    pairs = df["Channel Pair"].unique()
    results = []

    # Step 1: Collect all p-values for both comparisons
    p_values_real_rest = []
    p_values_real_imagined = []

    for pair in pairs:
        df_pair = df[df["Channel Pair"] == pair]
        pivot_df = df_pair.pivot(index="Subject", columns="Condition", values=metric)
        pivot_df = pivot_df.dropna()

        t_real_rest, p_real_rest = ttest_rel(pivot_df["Real"], pivot_df["Rest"])
        t_real_imagined, p_real_imagined = ttest_rel(pivot_df["Real"], pivot_df["Imagined"])

        results.append({
            "Channel Pair": pair,
            "t Real vs Rest": t_real_rest,
            "p Real vs Rest": p_real_rest,
            "t Real vs Imagined": t_real_imagined,
            "p Real vs Imagined": p_real_imagined
        })

        p_values_real_rest.append(p_real_rest)
        p_values_real_imagined.append(p_real_imagined)

    # Step 2: Apply Bonferroni correction
    _, p_real_rest_corr, _, _ = multipletests(p_values_real_rest, alpha=alpha, method='bonferroni')
    _, p_real_imagined_corr, _, _ = multipletests(p_values_real_imagined, alpha=alpha, method='bonferroni')

    # Step 3: Merge corrected p-values back into results
    for i, res in enumerate(results):
        res["p Real vs Rest (Bonf)"] = p_real_rest_corr[i]
        res["p Real vs Imagined (Bonf)"] = p_real_imagined_corr[i]

    df_stats = pd.DataFrame(results)
    return df_stats


def paired_ttest_plv(df, metric="PLV Mean"):
    """
    Run paired t-tests for PLV between conditions on each channel pair.
    """
    pairs = df["Channel Pair"].unique()
    results = []

    for pair in pairs:
        df_pair = df[df["Channel Pair"] == pair]
        pivot_df = df_pair.pivot(index="Subject", columns="Condition", values=metric)
        pivot_df = pivot_df.dropna()

        t_real_rest, p_real_rest = ttest_rel(pivot_df["Real"], pivot_df["Rest"])
        t_real_imagined, p_real_imagined = ttest_rel(pivot_df["Real"], pivot_df["Imagined"])

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
    """
    for idx, row in df_stats.iterrows():
        print(f"\n📊 Channel Pair: {row['Channel Pair']}")

        # Real vs Rest
        p1 = row['p Real vs Rest']
        color1 = "\033[92m" if p1 <= 0.05 else "\033[90m"
        print(f"{color1}   Real vs Rest: t = {row['t Real vs Rest']:.2f}, p = {p1:.4f} \033[0m")

        # Real vs Imagined
        p2 = row['p Real vs Imagined']
        color2 = "\033[92m" if p2 <= 0.05 else "\033[90m"
        print(f"{color2}   Real vs Imagined: t = {row['t Real vs Imagined']:.2f}, p = {p2:.4f} \033[0m")


# ------------------------------------------------------------------------------------- DECODING PIPELINE -------------------------------------------------------------------------------------

def perform_group_csp_lda(X, y, groups, title="Group Classification"):
    """
    Perform CSP+LDA classification with subject-aware cross-validation
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
    Perform CSP+LDA classification
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
                # Concatenate epochs for this subject
                if not (condition1 in eeg_data[subject] and condition2 in eeg_data[subject]):
                    print(f"Skipping subject {subject}: missing condition data")
                    continue
                    
                epochs1 = mne.concatenate_epochs(eeg_data[subject][condition1])
                epochs2 = mne.concatenate_epochs(eeg_data[subject][condition2])
                
                # Apply bandpass filter
                fmin, fmax = freq_band
                epochs1 = epochs1.copy().filter(fmin, fmax)
                epochs2 = epochs2.copy().filter(fmin, fmax)
                
                # Select motor channels
                motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'FC1', 'FC2', 'CP1', 'CP2']
                available_channels = [ch for ch in motor_channels if ch in epochs1.ch_names]
                
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
                
                # Simple undersampling - keep all trials from condition1 (real_right_hand)
                # and subsample condition2 (rest/imagined) to match
                if len(X2) > len(X1):
                    # Randomly sample from condition2 to match condition1 count
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
                
                # IMPORTANT: Z-score normalization per subject before combining
                # This helps address between-subject variability
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
                    if subject in eeg_data and len(eeg_data[subject][condition1]) > 0:
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
                    'results': perform_group_csp_lda(X_group, y_group, subject_group, 
                                                    f"Group: {condition1} vs {condition2}")
                }
                
                # Also calculate the average of individual accuracies for comparison
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


# ------------------------------------------------------------------------------------- FREQUENCY BAND ANALYSIS  -------------------------------------------------------------------------------------

def analyze_frequency_bands(eeg_data, subjects):
    """
    Analyze classification performance across different frequency bands
    within the pre-filtered range of 6-30 Hz.
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
        band_results['real_vs_rest'].append(
            group_results['real_right_hand_vs_rest']['individual_mean_acc'])
        band_results['real_vs_imagined'].append(
            group_results['real_right_hand_vs_imagined_right_hand']['individual_mean_acc'])
    
    # Visualize results
    plot_frequency_band_results(band_results, band_names)
    
    return band_results, band_names

def plot_frequency_band_results(band_results, band_names):
    """
    Create a bar plot comparing classification performance across frequency bands.
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
    best_real_rest = np.argmax(band_results['real_vs_rest'])
    best_real_imag = np.argmax(band_results['real_vs_imagined'])
    
    plt.annotate(f"Best: {band_results['real_vs_rest'][best_real_rest]:.3f}", 
                xy=(best_real_rest - width/2, band_results['real_vs_rest'][best_real_rest]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', color='#3498db', fontweight='bold')
    
    plt.annotate(f"Best: {band_results['real_vs_imagined'][best_real_imag]:.3f}", 
                xy=(best_real_imag + width/2, band_results['real_vs_imagined'][best_real_imag]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', color='#e74c3c', fontweight='bold')
    
    plt.show()
    
    # Print summary of results
    print("\n" + "="*80)
    print("FREQUENCY BAND ANALYSIS SUMMARY:")
    print("="*80)
    print(f"Best frequency band for Real vs. Rest: {band_names[best_real_rest]} "
          f"({band_results['real_vs_rest'][best_real_rest]:.3f})")
    print(f"Best frequency band for Real vs. Imagined: {band_names[best_real_imag]} "
          f"({band_results['real_vs_imagined'][best_real_imag]:.3f})")
    
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
    
    print(f"Real vs. Rest best in {band_names[best_real_rest]}: {interpretation[band_names[best_real_rest]]}")
    print(f"Real vs. Imagined best in {band_names[best_real_imag]}: {interpretation[band_names[best_real_imag]]}")


# ------------------------------------------------------------------------------------- TIME WINDOW ANALYSIS  -------------------------------------------------------------------------------------

def analyze_time_windows(eeg_data, subjects):
    """
    Analyze classification performance across different time windows
    within the epoch range from -1 to 3 seconds.
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
        
        # Store results
        window_results['real_vs_rest'].append(
            group_results['real_right_hand_vs_rest']['individual_mean_acc'])
        window_results['real_vs_imagined'].append(
            group_results['real_right_hand_vs_imagined_right_hand']['individual_mean_acc'])
    
    # Visualize results
    plot_time_window_results(window_results, window_names)
    
    return window_results, window_names

def classify_condition_pairs_with_time_window(eeg_data, subjects, time_window=None, freq_band=(8, 30)):
    """
    Modified version of classify_condition_pairs that uses a specific time window.
    This is a modified version of your existing function.
    """
    # This function would be a copy of your current classify_condition_pairs function
    # with a modification to use the specific time window during data extraction
    
    # Inside your original function, modify these lines:
    # times = epochs1_motor.times
    # start_idx = np.where(times >= 0)[0][0]  # Change this to time_window[0]
    # end_idx = np.where(times >= 2.0)[0][0]  # Change this to time_window[1]
    
    # Change to:
    # start_idx = np.where(times >= time_window[0])[0][0]
    # end_idx = np.where(times >= time_window[1])[0][0] if any(times >= time_window[1]) else -1
    
    # For brevity, I'm not copying the entire function here,
    # but you would take your existing classify_condition_pairs function
    # and modify it to accept and use a time_window parameter
    
    # This is a placeholder for demonstration - you'll implement the actual function
    pass

def plot_time_window_results(window_results, window_names):
    """
    Create a line plot showing how classification performance changes across time windows.
    """
    plt.figure(figsize=(12, 7))
    
    # Set up positions
    x = np.arange(len(window_names))
    
    # Plot lines with markers
    plt.plot(x, window_results['real_vs_rest'], 'o-', 
             linewidth=2, markersize=8, label='Real vs. Rest', color='#3498db')
    plt.plot(x, window_results['real_vs_imagined'], 's-', 
             linewidth=2, markersize=8, label='Real vs. Imagined', color='#e74c3c')
    
    # Add chance level line
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    
    # Labels and formatting
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Time Window (seconds)')
    plt.title('Classification Performance Across Time Windows')
    plt.xticks(x, window_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line at movement onset (t=0)
    plt.axvline(x=2, color='green', linestyle=':', 
                label='Movement Onset', alpha=0.7)
    
    plt.tight_layout()
    
    # Highlight the best performing time windows
    best_real_rest = np.argmax(window_results['real_vs_rest'])
    best_real_imag = np.argmax(window_results['real_vs_imagined'])
    
    plt.annotate(f"Best: {window_results['real_vs_rest'][best_real_rest]:.3f}", 
                xy=(best_real_rest, window_results['real_vs_rest'][best_real_rest]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', color='#3498db', fontweight='bold')
    
    plt.annotate(f"Best: {window_results['real_vs_imagined'][best_real_imag]:.3f}", 
                xy=(best_real_imag, window_results['real_vs_imagined'][best_real_imag]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', color='#e74c3c', fontweight='bold')
    
    plt.show()
    
    # Print summary of results
    print("\n" + "="*80)
    print("TIME WINDOW ANALYSIS SUMMARY:")
    print("="*80)
    print(f"Best time window for Real vs. Rest: {window_names[best_real_rest]} "
          f"({window_results['real_vs_rest'][best_real_rest]:.3f})")
    print(f"Best time window for Real vs. Imagined: {window_names[best_real_imag]} "
          f"({window_results['real_vs_imagined'][best_real_imag]:.3f})")
    
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
    
    print(f"Real vs. Rest best in {window_names[best_real_rest]}: {interpretation[window_names[best_real_rest]]}")
    print(f"Real vs. Imagined best in {window_names[best_real_imag]}: {interpretation[window_names[best_real_imag]]}")




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
    individual_results : dict
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
                # Concatenate epochs for this subject
                if not (condition1 in eeg_data[subject] and condition2 in eeg_data[subject]):
                    print(f"Skipping subject {subject}: missing condition data")
                    continue
                    
                epochs1 = mne.concatenate_epochs(eeg_data[subject][condition1])
                epochs2 = mne.concatenate_epochs(eeg_data[subject][condition2])
                
                # Apply bandpass filter
                fmin, fmax = freq_band
                epochs1 = epochs1.copy().filter(fmin, fmax)
                epochs2 = epochs2.copy().filter(fmin, fmax)
                
                # Select motor channels
                motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'FC1', 'FC2', 'CP1', 'CP2']
                available_channels = [ch for ch in motor_channels if ch in epochs1.ch_names]
                
                # Pick channels
                epochs1_motor = epochs1.copy().pick_channels(available_channels)
                epochs2_motor = epochs2.copy().pick_channels(available_channels)
                
                # Get time indices - THIS IS THE KEY CHANGE
                times = epochs1_motor.times
                
                if time_window is None:
                    # Default: use execution phase
                    start_idx = np.where(times >= 0)[0][0]
                    end_idx = np.where(times >= 2.0)[0][0] if any(times >= 2.0) else -1
                else:
                    # Use specified time window
                    start_time, end_time = time_window
                    start_idx = np.where(times >= start_time)[0][0]
                    end_idx = np.where(times >= end_time)[0][0] if any(times >= end_time) else -1
                
                # Extract data with specified time window
                X1 = epochs1_motor.get_data()[:, :, start_idx:end_idx]
                X2 = epochs2_motor.get_data()[:, :, start_idx:end_idx]
                
                # Simple undersampling - keep all trials from condition1 (real_right_hand)
                # and subsample condition2 (rest/imagined) to match
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
                
                # IMPORTANT: Z-score normalization per subject before combining
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
                    if subject in eeg_data and len(eeg_data[subject][condition1]) > 0:
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
                    'results': perform_group_csp_lda(X_group, y_group, subject_group, 
                                                    f"Group: {condition1} vs {condition2}")
                }
                
                # Also calculate the average of individual accuracies for comparison
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

