

def load_and_preprocess_subject(subject, runs_dict, l_freq=1., h_freq=40.):   # bandpass: (7.0, 30.0)
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
        Dict with keys 'rest', 'motor_execution', 'motor_imagery', each containing a raw object
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



def quick_plot(raw, title="Raw EEG Debug"):
    """
    Quick plot for sanity check.
    """
    raw.plot(n_channels=8, scalings="auto", title=title, show=True)



def extract_clean_epochs(raw, tmin=0.0, tmax=4.0, reject_boundary_epochs=True):
    
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


def compute_dfa_from_epochs(epochs, picks=None, dfa_window_sizes=np.logspace(4, 8, num=20, base=2, dtype=int)):
    """
    Compute DFA alpha values from MNE Epochs object.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object to extract data from.
    picks : list or None
        Channel names or indices to include. If None, uses all.
    dfa_window_sizes : array
        Array of window sizes (in samples) to use for DFA.
    
    Returns:
    --------
    alpha_vals : dict
        Dictionary with channel name as key and DFA alpha as value.
    """

    data = epochs.get_data(picks=picks)  # shape: (n_epochs, n_channels, n_times)
    ch_names = epochs.info['ch_names'] if picks is None else picks
    alpha_vals = {}

    for i, ch in enumerate(ch_names):
        signal = data[:, i, :].reshape(-1)  # Concatenate all epochs for that channel
        alpha = nolds.dfa(signal, fit_trend="poly", nvals=dfa_window_sizes)
        alpha_vals[ch] = alpha

    return alpha_vals


def compute_dfa(signal, nvals=None, order=1, debug_plot=True):
    """
    Perform Detrended Fluctuation Analysis (DFA) on a 1D signal.

    Parameters:
    -----------
    signal : np.ndarray
        1D array of EEG signal.
    nvals : list or array
        List of window sizes to test. If None, it will be auto-generated.
    order : int
        Order of the polynomial fit in each window (default: 1 → linear).
    debug_plot : bool
        Whether to plot the log-log curve and fitted line.

    Returns:
    --------
    alpha : float
        Estimated DFA exponent.
    """
    signal = np.array(signal)
    N = len(signal)

    # Step 1: Integrate the signal
    x = signal - np.mean(signal)
    y = np.cumsum(x)

    # Step 2: Define window sizes
    if nvals is None:
        nvals = np.unique(np.logspace(np.log10(10), np.log10(N // 4), num=20, dtype=int))
    
    fluctuations = []

    for n in nvals:
        if n >= N:
            continue

        num_segments = N // n
        local_flucts = []

        for i in range(num_segments):
            start = i * n
            end = start + n
            segment = y[start:end]

            # Fit polynomial and subtract
            t = np.arange(n)
            coeffs = np.polyfit(t, segment, order)
            fit = np.polyval(coeffs, t)
            detrended = segment - fit

            # RMS fluctuation
            local_flucts.append(np.sqrt(np.mean(detrended**2)))

        # Average over segments
        F_n = np.sqrt(np.mean(np.array(local_flucts) ** 2))
        fluctuations.append(F_n)

    # Step 3: Fit log-log curve
    log_n = np.log10(nvals[:len(fluctuations)])
    log_F = np.log10(fluctuations)
    slope, intercept, r, p, stderr = linregress(log_n, log_F)

    if debug_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(log_n, log_F, 'o-', label=f"DFA α = {slope:.3f}")
        plt.xlabel("log10(window size)")
        plt.ylabel("log10(RMS fluctuation)")
        plt.title("Detrended Fluctuation Analysis (DFA)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return slope

def compute_Fn(segment, n):
    """
    Compute the fluctuation function for a given window size n.
    """
    Xn = np.lib.stride_tricks.sliding_window_view(segment, window_shape=n)
    n_segments = Xn.shape[0]
    Fn_array = np.zeros(n_segments)

    t = np.arange(n)
    for i, seg in enumerate(Xn):
        trend = np.polyval(np.polyfit(t, seg, 1), t)
        detrended = seg - trend
        Fn_array[i] = np.std(detrended)

    return np.mean(Fn_array)


def compute_F(X, segment_sizes):
    """
    Computes the fluctuation function F(n) for each segment size in segment_sizes.
    X shape: (n_channels, n_times)
    """
    n_channels = X.shape[0]
    F = np.zeros((n_channels, len(segment_sizes)))

    # Integrate signal (DFA Step 1)
    X_int = np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)

    for ch in tqdm(range(n_channels), desc="Computing F(n) per channel"):
        for i, n in enumerate(segment_sizes):
            F[ch, i] = compute_Fn(X_int[ch], n)

    return F.mean(axis=0)  # Average across channels


def compute_DFA(segment_sizes, F, fitting_range):
    """
    Fit a linear regression in log-log space to estimate the DFA alpha.
    """
    idx_fit = (segment_sizes > fitting_range[0]) & (segment_sizes < fitting_range[1])
    log_n = np.log(segment_sizes[idx_fit])
    log_F = np.log(F[idx_fit])

    alpha, intercept = np.polyfit(log_n, log_F, 1)
    return alpha, intercept, log_n, log_F


def plot_dfa_fit(segment_sizes, F, alpha, intercept, fitting_range, freq=10, sfreq=160):
    """
    Visualize DFA result with log-log plot and fit.
    """
    N_cycles = (segment_sizes * freq) / sfreq
    idx_size = (segment_sizes > fitting_range[0]) & (segment_sizes < fitting_range[1])
    N_cycles_fit = N_cycles[idx_size]
    F_fit = np.exp(intercept) * segment_sizes[idx_size] ** alpha

    plt.figure(figsize=(8, 5))
    plt.loglog(N_cycles, F, 'o-', label="F(n)", markersize=5)
    plt.loglog(N_cycles_fit, F_fit, '--', label=f"Fit (α = {alpha:.3f})", linewidth=2)

    plt.axvline((fitting_range[0] * freq) / sfreq, color="gray", linestyle="--")
    plt.axvline((fitting_range[1] * freq) / sfreq, color="gray", linestyle="--", label="Fit Range")

    plt.xlabel("Number of Cycles")
    plt.ylabel("Fluctuation Function F(n)")
    plt.title("DFA Log-Log Scaling")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def extract_amplitude_envelope(signal, sfreq, band=(13, 30), filter_order=4):
    """
    Bandpass filters the signal and extracts the amplitude envelope using the Hilbert transform.

    Parameters
    ----------
    signal : 1D numpy array
        EEG time series (1 channel).
    sfreq : float
        Sampling frequency.
    band : tuple
        Frequency range for bandpass filter (e.g., (13, 30) for beta).
    filter_order : int
        Order of the Butterworth filter.

    Returns
    -------
    envelope : 1D numpy array
        Amplitude envelope of the band-limited signal.
    """
    nyquist = sfreq / 2
    low, high = band[0] / nyquist, band[1] / nyquist

    # Bandpass filter
    b, a = butter(filter_order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    # Hilbert transform
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    return envelope


def run_band_limited_dfa(epochs, channel="Cz", band=(13, 30), nvals=None):
    """
    Extracts the band-limited envelope from an EEG channel and runs DFA.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    channel : str
        Name of the EEG channel to use.
    band : tuple
        Frequency band (Hz).
    nvals : array or None
        DFA window sizes (optional).

    Returns
    -------
    alpha : float
        DFA exponent from the amplitude envelope.
    """
    sfreq = epochs.info['sfreq']
    signal = epochs.copy().pick(channel).get_data().reshape(-1)

    # Extract envelope
    envelope = extract_amplitude_envelope(signal, sfreq, band=band)

    # Compute DFA
    alpha = nolds.dfa(envelope, nvals=nvals)

    return alpha

def extract_wavelet_envelope(epochs, channel, sfreq, freq=13, n_cycles=7):
    """
    Extracts the amplitude envelope from wavelet transform (single frequency).

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    channel : str
        Channel name (e.g., 'Cz').
    sfreq : float
        Sampling frequency.
    freq : float
        Frequency of interest (e.g., 13 Hz for alpha).
    n_cycles : int
        Number of wavelet cycles.

    Returns
    -------
    envelope : 1D array
        Amplitude envelope over time (concatenated across epochs).
    """
    signal = epochs.copy().pick(channel).get_data()  # shape: (n_epochs, 1, n_times)
    power = tfr_array_morlet(
        data=signal,
        sfreq=sfreq,
        freqs=[freq],
        n_cycles=n_cycles,
        output='complex'
    )
    envelope = np.abs(power[0, 0])  # shape: (n_times,)

    return envelope

def compute_fluctuation_function(signal, segment_sizes):
    """
    Compute fluctuation function F(n) for DFA.

    Parameters
    ----------
    signal : 1D array
        Input signal (amplitude envelope).
    segment_sizes : array
        List of window sizes (samples)

    Returns
    -------
    F : array
        Fluctuation values for each window size.
    """
    signal = signal - np.mean(signal)
    integrated = np.cumsum(signal)

    F = []
    for size in segment_sizes:
        n_segments = len(integrated) // size
        if n_segments < 2:
            continue

        segments = integrated[:n_segments * size].reshape(n_segments, size)
        fluctuations = []
        for seg in segments:
            t = np.arange(size)
            trend = np.polyval(np.polyfit(t, seg, 1), t)
            detrended = seg - trend
            fluctuations.append(np.std(detrended))

        F.append(np.mean(fluctuations))

    return np.array(F)


def compute_dfa_scaling(segment_sizes, F, fit_range=(10, 1000)):
    idx = (segment_sizes >= fit_range[0]) & (segment_sizes <= fit_range[1])
    log_n = np.log10(segment_sizes[idx])
    log_F = np.log10(F[idx])
    slope, intercept = np.polyfit(log_n, log_F, 1)
    return slope, intercept
