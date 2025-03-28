"""
Visualization functions for EEG data analysis of motor movement and imagery.
Provides plotting functions for raw data, ERD/ERS, connectivity, and classification results.
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MNE for EEG-specific functions
import mne

# Import custom functions
import my_functions as my_fun


# -----------------------------------------------------------------------------
# RAW DATA VISUALIZATION
# -----------------------------------------------------------------------------

def plot_raw(raw, title="EEG Signal", n_channels=10):
    """
    Plot raw EEG data.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    title : str
        Plot title
    n_channels : int
        Number of channels to plot
    """
    data, times = raw[:, :]
    plt.figure(figsize=(12, 6))
    
    for i in range(min(n_channels, data.shape[0])):
        plt.plot(times, data[i] * 1e6 + i * 100, label=raw.ch_names[i])  # µV scale with offset
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Amplitude (µV)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_raw_eeg(epochs, subject, condition, n_channels=10):
    """
    Plot epoched EEG data.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    subject : str
        Subject identifier
    condition : str
        Experimental condition
    n_channels : int
        Number of channels to plot
    """
    # Get data from first epoch, limit to first n_channels, convert to µV
    raw = epochs.get_data()[:1, :n_channels, :] * 1e6
    times = epochs.times
    
    plt.figure(figsize=(12, 6))
    
    for i in range(raw.shape[1]):
        # Plot with vertical offset for readability
        plt.plot(times, raw[0, i, :] + i * 20, label=f"{epochs.ch_names[i]}")
    
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Amplitude (µV)")
    plt.title(f"Raw EEG Signal - {subject} ({condition})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_erd_ers(epochs, subject, condition, fmin=6, fmax=30, motor_channels=None):
    """
    Plot Event-Related Desynchronization/Synchronization for motor channels.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    subject : str
        Subject identifier
    condition : str
        Experimental condition
    fmin, fmax : float
        Frequency range for ERD/ERS calculation
    motor_channels : list
        List of motor channels to plot (default: predefined motor channels)
    """
    # Compute ERD/ERS using function from my_functions
    erd_ers, times, freqs = my_fun.compute_erd_ers(epochs, fmin=fmin, fmax=fmax)
    available_channels = epochs.ch_names
    
    # Define default motor channels if not provided
    if motor_channels is None:
        motor_channels = ["C3", "C4", "Cz", "C1", "C2", "CP3", "CP4", "FC3", "FC4"]
    
    # Find which motor channels are available in the data
    matched_channels = [ch for ch in motor_channels if ch in available_channels]
    
    if not matched_channels:
        print("⚠️ No valid motor channels found in dataset! Available channels:")
        print(", ".join(available_channels))
        return
    
    plt.figure(figsize=(12, 7))
    
    for ch in matched_channels:
        ch_idx = available_channels.index(ch)
        
        # Create frequency mask for the specified range
        freq_mask = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Average over epochs and selected frequencies
        erd_curve = np.mean(erd_ers[:, ch_idx, freq_mask, :], axis=(0, 1))
        
        plt.plot(times, erd_curve, label=f"{ch}", linewidth=2)
    
    # Add movement onset line
    plt.axvline(0, color='k', linestyle='--', label="Movement Onset")
    
    # Add baseline shading
    plt.axvspan(-1, 0, color='lightgray', alpha=0.3, label="Baseline")
    
    # Add zero line (no change from baseline)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel("Time (s)")
    plt.ylabel("ERD/ERS (% Change from Baseline)")
    plt.title(f"ERD/ERS in {subject} ({condition}) [{fmin}-{fmax} Hz]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CONNECTIVITY VISUALIZATION
# -----------------------------------------------------------------------------

def plot_plv_matrix(plv_matrix, ch_names, title="PLV Matrix", figsize=(10, 8)):
    """
    Plot Phase Locking Value matrix as a heatmap.
    
    Parameters:
    -----------
    plv_matrix : ndarray
        PLV matrix (n_channels x n_channels)
    ch_names : list
        List of channel names
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(plv_matrix, interpolation="nearest", origin="lower", 
                   cmap="viridis", vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Phase Locking Value')
    
    # Add channel labels
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.yticks(np.arange(len(ch_names)), ch_names)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_plv_difference(plv_a, plv_b, ch_names, title="PLV Difference", figsize=(10, 8)):
    """
    Plot the difference between two PLV matrices as a heatmap.
    
    Parameters:
    -----------
    plv_a, plv_b : ndarray
        PLV matrices to compare
    ch_names : list
        List of channel names
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    diff = plv_a - plv_b
    
    # Set symmetric color limits for difference plot
    v_abs_max = np.max(np.abs(diff))
    
    plt.figure(figsize=figsize)
    im = plt.imshow(diff, interpolation="nearest", origin="lower", 
                   cmap="bwr", vmin=-v_abs_max, vmax=v_abs_max)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('PLV Difference')
    
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90)
    plt.yticks(np.arange(len(ch_names)), ch_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_motor_plv_difference(plv_a, plv_b, ch_names, motor_chs=None, 
                            title="Motor Cortex PLV Difference", figsize=(8, 7)):
    """
    Plot the PLV difference for motor cortex channels only.
    
    Parameters:
    -----------
    plv_a, plv_b : ndarray
        PLV matrices to compare
    ch_names : list
        List of channel names
    motor_chs : list
        List of motor channel names to include
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    # Define default motor channels if not provided
    if motor_chs is None:
        motor_chs = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "C1", "C2"]

    # Find indices of motor channels present in data
    indices = [ch_names.index(ch) for ch in motor_chs if ch in ch_names]
    if not indices:
        print("⚠️ No motor channels found in the data!")
        print(f"Available channels: {', '.join(ch_names)}")
        return
        
    motor_ch_names = [ch_names[i] for i in indices]

    # Extract sub-matrices for motor channels using numpy's advanced indexing
    motor_plv_a = plv_a[np.ix_(indices, indices)]
    motor_plv_b = plv_b[np.ix_(indices, indices)]

    # Compute difference
    diff = motor_plv_a - motor_plv_b
    v_abs_max = np.max(np.abs(diff))

    # Plot
    plt.figure(figsize=figsize)
    im = plt.imshow(diff, interpolation="nearest", origin="lower", 
                   cmap="bwr", vmin=-v_abs_max, vmax=v_abs_max)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('PLV Difference')
    
    plt.xticks(np.arange(len(motor_ch_names)), motor_ch_names, rotation=90)
    plt.yticks(np.arange(len(motor_ch_names)), motor_ch_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_plv_coherence(df, metric="PLV Mean", figsize=(14, 7)):
    """
    Plot PLV or Coherence comparisons between multiple conditions.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing PLV & Coherence results
    metric : str
        Which metric to plot: "PLV Mean", "PLV Max", "Coherence Mean", or "Coherence Max"
    figsize : tuple
        Figure size
    """
    if metric not in ["PLV Mean", "PLV Max", "Coherence Mean", "Coherence Max"]:
        raise ValueError("Metric must be one of: 'PLV Mean', 'PLV Max', 'Coherence Mean', 'Coherence Max'")
    
    plt.figure(figsize=figsize)
    
    # Generate color palette based on number of conditions
    n_conditions = df["Condition"].nunique()
    palette = sns.color_palette("tab10", n_colors=n_conditions)
    
    # Create grouped bar plot
    ax = sns.barplot(data=df, x="Channel Pair", y=metric, hue="Condition", palette=palette)
    
    # Add data values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    # Set labels and title
    metric_type = "Phase Locking Value" if "PLV" in metric else "Coherence"
    measure_type = "Mean" if "Mean" in metric else "Maximum"
    
    plt.xlabel("Electrode Pair")
    plt.ylabel(f"{metric_type} ({measure_type})")
    plt.title(f"{metric_type} {measure_type} Comparison Across Conditions")
    
    plt.legend(title="Condition")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLASSIFICATION RESULTS VISUALIZATION
# -----------------------------------------------------------------------------

def visualize_comparison(group_results, individual_results):
    """
    Visualize and compare classification results between real vs. rest and real vs. imagined.
    
    Parameters:
    -----------
    group_results : dict
        Dictionary containing group-level results from classify_condition_pairs
    individual_results : dict
        Dictionary containing individual subject results
        
    Returns:
    --------
    summary : str
        Text summary of classification results
    """
    # Check if we have both condition pairs
    pairs = ['real_right_hand_vs_rest', 'real_right_hand_vs_imagined_right_hand']
    if not all(pair in group_results for pair in pairs):
        raise ValueError("Both condition pairs must be in group_results")
    
    # Extract results
    real_vs_rest = group_results['real_right_hand_vs_rest']
    real_vs_imagined = group_results['real_right_hand_vs_imagined_right_hand']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. ROC Curve Comparison
    ax1 = plt.subplot2grid((3, 3), (0, 0))
    
    # Plot both ROC curves
    ax1.plot(
        real_vs_rest['results']['fpr'], 
        real_vs_rest['results']['tpr'], 
        lw=2, 
        label=f"Real vs. Rest (AUC = {real_vs_rest['results']['roc_auc']:.2f})"
    )
    ax1.plot(
        real_vs_imagined['results']['fpr'], 
        real_vs_imagined['results']['tpr'], 
        lw=2, 
        label=f"Real vs. Imagined (AUC = {real_vs_imagined['results']['roc_auc']:.2f})"
    )
    
    # Add reference diagonal
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves Comparison')
    ax1.legend(loc="lower right")
    
    # 2. Accuracy Comparison (Group vs Individual Average)
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    
    # Prepare data
    conditions = ['Real vs. Rest', 'Real vs. Imagined']
    
    # Group accuracies (subject-aware CV)
    group_accuracies = [
        real_vs_rest['results']['accuracy'],
        real_vs_imagined['results']['accuracy']
    ]
    group_errors = [
        real_vs_rest['results']['std'],
        real_vs_imagined['results']['std']
    ]
    
    # Average of individual subject accuracies
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
    
    # Plot bars
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
    
    # Add reference line for chance level
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
    
    # Labels and formatting
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy Comparison: Group vs. Individual Average')
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    
    # 3. Confusion Matrix - Real vs. Rest
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
    
    # 4. Confusion Matrix - Real vs. Imagined
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
    
    # 5. CSP Pattern Visualization
    ax5 = plt.subplot2grid((3, 3), (1, 2))
    try:
        # Extract CSP patterns from pipeline
        csp_real_rest = real_vs_rest['results']['pipeline'].named_steps['csp']
        info = real_vs_rest['info']
        channels = real_vs_rest['channels']
        
        # Create reduced info object with only the channels used in CSP
        info_reduced = mne.pick_info(
            info.copy(), 
            [info['ch_names'].index(ch) for ch in channels if ch in info['ch_names']]
        )
        
        # Get first pattern (most discriminative)
        pattern_data = csp_real_rest.patterns_[:, 0]
        
        # Plot topographic map
        mne.viz.plot_topomap(
            pattern_data, 
            info_reduced, 
            axes=ax5, 
            show=False
        )
        ax5.set_title('Most Discriminative Pattern\nReal vs. Rest')
    except Exception as e:
        print(f"Error plotting CSP patterns: {e}")
        ax5.text(0.5, 0.5, "Error plotting CSP pattern", 
                 ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Individual Subject Performance
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Get individual subject accuracies
    subjects = sorted(individual_results['real_right_hand_vs_rest'].keys())
    real_rest_accs = [individual_results['real_right_hand_vs_rest'][subj]['accuracy'] 
                      for subj in subjects]
    
    # Only include subjects with both conditions
    valid_subjects = []
    valid_real_rest = []
    valid_real_imag = []
    
    for i, subj in enumerate(subjects):
        if subj in individual_results['real_right_hand_vs_imagined_right_hand']:
            valid_subjects.append(subj)
            valid_real_rest.append(real_rest_accs[i])
            valid_real_imag.append(
                individual_results['real_right_hand_vs_imagined_right_hand'][subj]['accuracy']
            )
    
    # Set up positions for bars
    x = np.arange(len(valid_subjects))
    width = 0.35
    
    # Plot individual accuracies
    ax6.bar(
        x - width/2, 
        valid_real_rest, 
        width,
        color='#3498db',
        label='Real vs. Rest'
    )
    ax6.bar(
        x + width/2, 
        valid_real_imag, 
        width,
        color='#e74c3c',
        label='Real vs. Imagined'
    )
    
    # Add reference line
    ax6.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
    
    # Labels and formatting
    ax6.set_ylabel('Classification Accuracy')
    ax6.set_xlabel('Subject')
    ax6.set_title('Individual Subject Classification Performance')
    ax6.set_xticks(x)
    ax6.set_xticklabels(valid_subjects, rotation=45)
    ax6.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Group-Level Classification Results Comparison', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Create summary text of results
    summary = (
        f"SUMMARY OF CLASSIFICATION RESULTS:\n\n"
        f"Group-Level (Subject-Aware CV):\n"
        f"Real vs. Rest: Accuracy = {real_vs_rest['results']['accuracy']:.3f} ± {real_vs_rest['results']['std']:.3f}, "
        f"AUC = {real_vs_rest['results']['roc_auc']:.3f}\n"
        f"Real vs. Imagined: Accuracy = {real_vs_imagined['results']['accuracy']:.3f} ± {real_vs_imagined['results']['std']:.3f}, "
        f"AUC = {real_vs_imagined['results']['roc_auc']:.3f}\n\n"
        
        f"Average of Individual Subjects:\n"
        f"Real vs. Rest: Accuracy = {real_vs_rest['individual_mean_acc']:.3f} ± {real_vs_rest['individual_std_acc']:.3f}\n"
        f"Real vs. Imagined: Accuracy = {real_vs_imagined['individual_mean_acc']:.3f} ± {real_vs_imagined['individual_std_acc']:.3f}\n\n"
        
        f"Number of subjects with both classifications: {len(valid_subjects)}"
    )
    
    print("\n" + "="*80)
    print(summary)
    print("="*80)
    
    return summary


def plot_decoding_over_time(times, scores_real_rest, scores_real_imag, window_size=0.2):
    """
    Plot classification accuracy over time using sliding window decoding.
    
    Parameters:
    -----------
    times : ndarray
        Time points (center of each window)
    scores_real_rest : ndarray
        Classification scores for Real vs. Rest
    scores_real_imag : ndarray
        Classification scores for Real vs. Imagined
    window_size : float
        Size of the sliding window in seconds
    """
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy curves
    plt.plot(times, scores_real_rest, 'o-', label='Real vs. Rest', 
             color='#3498db', linewidth=2, markersize=5)
    plt.plot(times, scores_real_imag, 's-', label='Real vs. Imagined', 
             color='#e74c3c', linewidth=2, markersize=5)
    
    # Add reference line for chance level
    plt.axhline(0.5, color='k', linestyle='--', label='Chance Level')
    
    # Add vertical line at movement onset
    plt.axvline(0, color='g', linestyle='-', label='Movement Onset', alpha=0.7)
    
    # Shade baseline period
    plt.axvspan(-1, 0, color='lightgray', alpha=0.3, label='Baseline Period')
    
    # Shade significance regions (example - replace with actual significance)
    # plt.fill_between(times, 0.5, scores_real_rest, where=(scores_real_rest > 0.6),
    #                  color='#3498db', alpha=0.2)
    
    # Labels and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title(f'Classification Accuracy Over Time (Window Size: {window_size}s)')
    plt.legend(loc='upper right')
    
    # Set reasonable y-limits
    plt.ylim([0.4, 1.0])
    
    # Grid and formatting
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(confusion_matrices, condition_names, time_points):
    """
    Plot confusion matrices across multiple time points.
    
    Parameters:
    -----------
    confusion_matrices : list of ndarrays
        List of confusion matrices at different time points
    condition_names : list
        Names of the conditions (for labels)
    time_points : list
        Time points corresponding to each confusion matrix
    """
    n_times = len(time_points)
    fig, axes = plt.subplots(1, n_times, figsize=(n_times * 4, 4))
    
    if n_times == 1:
        axes = [axes]
    
    for i, (cm, time) in enumerate(zip(confusion_matrices, time_points)):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=condition_names,
            yticklabels=condition_names,
            ax=axes[i]
        )
        axes[i].set_xlabel('Predicted')
        if i == 0:
            axes[i].set_ylabel('True')
        axes[i].set_title(f'Time: {time:.1f}s')
    
    plt.tight_layout()
    plt.suptitle('Confusion Matrices Over Time', y=1.05)
    plt.show()