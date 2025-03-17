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




