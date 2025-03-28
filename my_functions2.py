"""
Functions for EEG preprocessing, analysis, and fractal features extraction.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import linregress
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------

def load_and_preprocess_subject(subject, runs_dict, l_freq=1., h_freq=40.):
    """
    Load & preprocess EEGBCI data for a subject split into conditions.

    Parameters:
    -----------
    subject : int
        Subject ID (e.g., 1)
    runs_dict : dict
        Dictionary like {'rest': [1], 'motor_execution': [3,7,11], 'motor_imagery': [4,8,12]}
    l_freq : float
        Bandpass low cutoff
    h_freq : float
        Bandpass high cutoff
    
    Returns:
    --------
    subject_data : dict
        Dict with keys matching runs_dict, each containing a raw object
    """
    subject_data = {}

    for condition, run_list in runs_dict.items():
        print(f"\n➡️ Loading {condition.upper()} | Runs: {run_list}")

        raw_fnames = eegbci.load_data(subject, run_list)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw_concat = concatenate_raws(raws)
        
        # Preprocessing pipeline
        eegbci.standardize(raw_concat)
        montage = make_standard_montage('standard_1005')
        raw_concat.set_montage(montage)
        raw_concat.set_eeg_reference(projection=True)
        raw_concat.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation="edge")
        
        print(f"✅ {condition} | Shape: {raw_concat._data.shape} | Duration: {raw_concat.times[-1] / 60:.2f} min")

        # Save preprocessed raw for this condition
        subject_data[condition] = raw_concat

    return subject_data


def extract_clean_epochs(raw, tmin=0.0, tmax=4.0, reject_boundary_epochs=True):
    """
    Extract clean epochs from a raw object, separating rest and task events.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data with events/annotations
    tmin, tmax : float
        Epoch time window around events
    reject_boundary_epochs : bool
        Whether to reject epochs that overlap with boundary annotations
        
    Returns:
    --------
    epochs_dict : dict
        Dictionary with 'rest' and 'task' keys containing respective epochs
    """
    events, event_id = mne.events_from_annotations(raw)
    print(f"\n⏺️ Used Annotations descriptions: {list(event_id.keys())}")

    rest_id = event_id.get("T0")
    task_ids = [event_id.get("T1"), event_id.get("T2")]
    task_ids = [ev_id for ev_id in task_ids if ev_id is not None]  # Filter out None

    # Create epochs for REST (T0) if it exists
    if rest_id is not None:
        epochs_rest = mne.Epochs(
            raw, events, event_id=rest_id,
            tmin=tmin, tmax=tmax, baseline=None,
            reject_by_annotation=reject_boundary_epochs,
            preload=True
        )
    else:
        epochs_rest = None
        print("⚠️ No T0 (REST) events found.")

    # Create epochs for TASK (T1 + T2) if available
    if task_ids:
        epochs_task = mne.Epochs(
            raw, events, event_id=task_ids,
            tmin=tmin, tmax=tmax, baseline=None,
            reject_by_annotation=reject_boundary_epochs,
            preload=True
        )
    else:
        epochs_task = None
        print("⚠️ No T1/T2 (TASK) events found.")

    print(f"📊 Extracted {len(epochs_rest) if epochs_rest else 0} REST epochs & {len(epochs_task) if epochs_task else 0} TASK epochs")

    return {"rest": epochs_rest, "task": epochs_task}


def quick_plot(raw, title="Raw EEG Debug", n_channels=8):
    """
    Quick plot for sanity check.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    title : str
        Plot title
    n_channels : int
        Number of channels to display
    """
    raw.plot(n_channels=n_channels, scalings="auto", title=title, show=True)


# -----------------------------------------------------------------------------
# Signal Processing Functions
# -----------------------------------------------------------------------------

def filter_signal(signal, sfreq, band, order=4):
    """
    Apply bandpass filter to signal.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    sfreq : float
        Sampling frequency (Hz)
    band : tuple
        (low_freq, high_freq) in Hz
    order : int
        Filter order
        
    Returns:
    --------
    filtered : ndarray
        Bandpass filtered signal
    """
    nyquist = sfreq / 2.0
    low, high = band[0] / nyquist, band[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_amplitude_envelope(signal, sfreq, band=(13, 30), filter_order=4):
    """
    Bandpass filters the signal and extracts the amplitude envelope using the Hilbert transform.

    Parameters:
    -----------
    signal : 1D numpy array
        EEG time series (1 channel)
    sfreq : float
        Sampling frequency
    band : tuple
        Frequency range for bandpass filter (e.g., (13, 30) for beta)
    filter_order : int
        Order of the Butterworth filter

    Returns:
    --------
    envelope : 1D numpy array
        Amplitude envelope of the band-limited signal
    """
    # Bandpass filter
    filtered = filter_signal(signal, sfreq, band, order=filter_order)
    
    # Hilbert transform
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    return envelope


def extract_wavelet_envelope(epochs, channel, freq=13, n_cycles=7):
    """
    Extracts the amplitude envelope from wavelet transform (single frequency).

    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object
    channel : str
        Channel name (e.g., 'Cz')
    freq : float
        Frequency of interest (e.g., 13 Hz for alpha)
    n_cycles : int
        Number of wavelet cycles

    Returns:
    --------
    envelope : 1D array
        Amplitude envelope over time (concatenated across epochs)
    """
    # Extract data and sampling frequency
    signal = epochs.copy().pick(channel).get_data()  # shape: (n_epochs, 1, n_times)
    sfreq = epochs.info['sfreq']
    
    # Apply Morlet wavelet transform
    power = mne.time_frequency.tfr_array_morlet(
        data=signal,
        sfreq=sfreq,
        freqs=[freq],
        n_cycles=n_cycles,
        output='complex'
    )
    
    # Get amplitude
    envelope = np.abs(power).squeeze()  # shape: (n_epochs, n_times)
    
    # Flatten across epochs if multiple epochs
    if envelope.ndim > 1:
        envelope = envelope.reshape(-1)
        
    return envelope


# -----------------------------------------------------------------------------
# Detrended Fluctuation Analysis (DFA)
# -----------------------------------------------------------------------------

def compute_dfa(signal, scales=None, order=1, fit_range=None, return_fit=False):
    """
    Perform Detrended Fluctuation Analysis on a signal.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal (1D array)
    scales : ndarray or None
        Array of window sizes in samples. If None, auto-generated.
    order : int
        Order of polynomial detrending (1=linear, 2=quadratic, etc.)
    fit_range : tuple or None
        (min_scale, max_scale) for fitting the scaling exponent.
        If None, use all scales.
    return_fit : bool
        If True, return fluctuation values and scales along with alpha.
    
    Returns:
    --------
    alpha : float
        DFA scaling exponent
    [fluctuations, scales] : list, optional
        Only returned if return_fit=True
    """
    # 1. Prepare the signal
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    y = np.cumsum(signal)  # Integration
    n_samples = len(y)
    
    # 2. Define scales if not provided
    if scales is None:
        min_scale = 10  # Minimum window size
        max_scale = n_samples // 4  # Maximum window size
        scales = np.unique(np.logspace(
            np.log10(min_scale), np.log10(max_scale), 20, dtype=int
        ))
    
    # 3. Compute fluctuations for each scale
    fluctuations = []
    used_scales = []
    
    for scale in scales:
        if scale >= n_samples:
            continue
        
        # Number of non-overlapping windows
        n_windows = n_samples // scale
        if n_windows < 1:
            continue
            
        used_scales.append(scale)
        
        # Create windows
        windows = y[:n_windows*scale].reshape((n_windows, scale))
        
        # Detrend each window
        t = np.arange(scale)
        window_fluctuations = []
        
        for window in windows:
            # Fit polynomial of specified order
            p = np.polyfit(t, window, order)
            fit = np.polyval(p, t)
            
            # Calculate fluctuation (root mean square)
            residuals = window - fit
            fluctuation = np.sqrt(np.mean(residuals**2))
            window_fluctuations.append(fluctuation)
        
        # Average fluctuation over all windows
        fluctuations.append(np.mean(window_fluctuations))
    
    # Convert to arrays
    fluctuations = np.array(fluctuations)
    used_scales = np.array(used_scales)
    
    if len(fluctuations) < 4:
        raise ValueError("Not enough scales for reliable DFA. Try a longer signal.")
        
    # 4. Fit the scaling exponent (alpha)
    if fit_range is not None:
        min_scale, max_scale = fit_range
        idx = (used_scales >= min_scale) & (used_scales <= max_scale)
        if np.sum(idx) < 4:
            print(f"Warning: Only {np.sum(idx)} points in fit range. Consider adjusting fit_range.")
        log_scales = np.log10(used_scales[idx])
        log_fluct = np.log10(fluctuations[idx])
    else:
        log_scales = np.log10(used_scales)
        log_fluct = np.log10(fluctuations)
    
    # Linear regression to get alpha
    slope, _, r_value, _, _ = linregress(log_scales, log_fluct)
    alpha = slope
    
    if return_fit:
        return alpha, fluctuations, used_scales
    else:
        return alpha


def compute_dfa_from_epochs(epochs, picks=None, band=None, order=1, fit_range=None, 
                          envelope_method='hilbert'):
    """
    Compute DFA scaling exponents from MNE Epochs.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object
    picks : list or None
        Channel selection (names or indices)
    band : tuple or None
        (low_freq, high_freq) band to filter signal. If None, uses raw signal.
    order : int
        Order of polynomial detrending
    fit_range : tuple or None
        (min_scale, max_scale) for fitting. If None, uses all scales.
    envelope_method : str
        Method to compute amplitude envelope if band is provided:
        'hilbert' or 'wavelet'
    
    Returns:
    --------
    alpha_values : dict
        Dictionary with channel names as keys and DFA alphas as values
    """
    # Get data and channel names
    if picks is None:
        picks = epochs.ch_names
    
    data = epochs.get_data(picks=picks)
    ch_names = epochs.ch_names if picks is None else picks
    
    if isinstance(ch_names, str):
        ch_names = [ch_names]
        
    # Extract signals and apply filtering if needed
    alpha_values = {}
    sfreq = epochs.info['sfreq']
    
    for i, ch_name in enumerate(ch_names):
        # Extract and flatten channel data
        signal = data[:, i, :].flatten()
        
        # Apply bandpass filtering if specified
        if band is not None:
            if envelope_method == 'hilbert':
                signal = extract_amplitude_envelope(signal, sfreq, band)
            elif envelope_method == 'wavelet':
                # Use central frequency of the band
                center_freq = (band[0] + band[1]) / 2
                signal = extract_wavelet_envelope(
                    epochs.copy().pick(ch_name), 
                    channel=ch_name, 
                    freq=center_freq
                )
            else:
                raise ValueError(f"Unknown envelope method: {envelope_method}")
        
        # Compute DFA
        alpha = compute_dfa(signal, order=order, fit_range=fit_range)
        alpha_values[ch_name] = alpha
        
    return alpha_values


def plot_dfa_result(fluctuations, scales, alpha, fit_range=None, ax=None, 
                  freq_band=None, sfreq=None, title=None):
    """
    Plot DFA results in log-log space.
    
    Parameters:
    -----------
    fluctuations : ndarray
        Fluctuation values from DFA
    scales : ndarray
        Window sizes used in DFA
    alpha : float
        Scaling exponent
    fit_range : tuple or None
        (min_scale, max_scale) range used for fitting
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new figure.
    freq_band : tuple or None
        Frequency band if analyzing an envelope (for labeling)
    sfreq : float or None
        Sampling frequency (to convert scales to seconds)
    title : str or None
        Custom title for the plot
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot fluctuations
    ax.loglog(scales, fluctuations, 'o-', label='Data')
    
    # Plot fit line
    if fit_range is not None:
        min_scale, max_scale = fit_range
        idx = (scales >= min_scale) & (scales <= max_scale)
        fit_scales = scales[idx]
        
        # Calculate fit line
        log_scales = np.log10(fit_scales)
        log_start = np.log10(fit_scales[0])
        first_point_idx = np.where(scales == fit_scales[0])[0][0]
        fit_line = 10**(np.log10(fluctuations[first_point_idx]) + 
                      alpha * (log_scales - log_start))
        
        # Plot fit line and range markers
        ax.loglog(fit_scales, fit_line, '--', linewidth=2, 
                 label=f'α = {alpha:.3f}')
        ax.axvline(min_scale, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(max_scale, color='gray', linestyle=':', alpha=0.7)
    else:
        # Use all scales for fit visualization
        log_scales = np.log10(scales)
        fit_line = 10**(np.log10(fluctuations[0]) + alpha * (log_scales - log_scales[0]))
        ax.loglog(scales, fit_line, '--', linewidth=2, 
                 label=f'α = {alpha:.3f}')
    
    # Create x-label based on available information
    if sfreq is not None:
        time_scales = scales / sfreq
        ax.set_xlabel('Window Size (seconds)')
        
        # Add second x-axis for sample counts
        ax2 = ax.twiny()
        ax2.loglog(scales, fluctuations, alpha=0)  # Invisible, just to match scales
        ax2.set_xlabel('Window Size (samples)')
    else:
        ax.set_xlabel('Window Size (samples)')
    
    # Create title based on available information
    if title is None:
        title = 'Detrended Fluctuation Analysis'
        if freq_band is not None:
            title = f'DFA - {freq_band[0]}-{freq_band[1]} Hz Band'
    
    # Formatting
    ax.set_ylabel('Fluctuation F(n)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.7)
    ax.legend()
    
    return ax


def analyze_band_dfa(epochs, channel, bands, fit_range=None, order=1, plot=True):
    """
    Perform DFA across multiple frequency bands for a single channel.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object
    channel : str
        Channel name to analyze
    bands : dict
        Dictionary of frequency bands, e.g., {'alpha': (8, 13), 'beta': (13, 30)}
    fit_range : tuple or None
        (min_scale, max_scale) for fitting
    order : int
        Order of polynomial detrending
    plot : bool
        Whether to plot the results
        
    Returns:
    --------
    results : dict
        Dictionary with band names as keys and DFA alphas as values
    """
    results = {}
    signal = epochs.copy().pick(channel).get_data().reshape(-1)
    sfreq = epochs.info['sfreq']
    
    if plot:
        fig, axes = plt.subplots(len(bands), 1, figsize=(10, 4*len(bands)))
        if len(bands) == 1:
            axes = [axes]
    
    for i, (band_name, freq_range) in enumerate(bands.items()):
        # Extract envelope
        envelope = extract_amplitude_envelope(signal, sfreq, freq_range)
        
        # Compute DFA
        alpha, fluctuations, scales = compute_dfa(
            envelope, fit_range=fit_range, order=order, return_fit=True
        )
        results[band_name] = alpha
        
        # Plot if requested
        if plot:
            plot_dfa_result(
                fluctuations, scales, alpha, fit_range=fit_range, ax=axes[i],
                freq_band=freq_range, sfreq=sfreq, 
                title=f"DFA - {band_name.capitalize()} Band ({freq_range[0]}-{freq_range[1]} Hz)"
            )
    
    if plot:
        plt.tight_layout()
        plt.show()
        
    return results


def compare_conditions_dfa(epochs_dict, channel, band=None, fit_range=None, order=1):
    """
    Compare DFA scaling exponents between conditions.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary with condition names as keys and mne.Epochs as values
    channel : str
        Channel name to analyze
    band : tuple or None
        Frequency band to analyze. If None, uses raw signal.
    fit_range : tuple or None
        (min_scale, max_scale) for fitting
    order : int
        Order of polynomial detrending
        
    Returns:
    --------
    results : dict
        Nested dictionary with conditions, values, and a comparison figure
    """
    results = {'alpha_values': {}}
    
    # Create figure for comparison
    fig, axes = plt.subplots(1, len(epochs_dict), figsize=(5*len(epochs_dict), 5))
    if len(epochs_dict) == 1:
        axes = [axes]
    
    # Analyze each condition
    for i, (condition, epochs) in enumerate(epochs_dict.items()):
        signal = epochs.copy().pick(channel).get_data().reshape(-1)
        sfreq = epochs.info['sfreq']
        
        # Apply filtering if band is specified
        if band is not None:
            signal = extract_amplitude_envelope(signal, sfreq, band)
            title = f"{condition.upper()} - {band[0]}-{band[1]} Hz"
        else:
            title = f"{condition.upper()} - Raw Signal"
        
        # Compute DFA
        alpha, fluctuations, scales = compute_dfa(
            signal, fit_range=fit_range, order=order, return_fit=True
        )
        results['alpha_values'][condition] = alpha
        
        # Plot
        plot_dfa_result(
            fluctuations, scales, alpha, fit_range=fit_range, ax=axes[i],
            freq_band=band, sfreq=sfreq, title=title
        )
    
    plt.tight_layout()
    results['figure'] = fig
    
    # Print summary
    print("\n" + "="*50)
    print("DFA SCALING EXPONENTS COMPARISON")
    print("="*50)
    for condition, alpha in results['alpha_values'].items():
        print(f"{condition.upper()}: α = {alpha:.3f}")
    
    return results