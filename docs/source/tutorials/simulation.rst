Simulation Utilities
====================

neuroim provides utilities for generating synthetic neuroimaging data, useful for method development, testing, and statistical power analysis.

Simulating fMRI Data
--------------------

Generate synthetic 4D fMRI datasets with configurable properties:

.. code-block:: python

    import neuroim
    import numpy as np

    # Create a basic simulated fMRI dataset
    sim_vec = neuroim.simulate_fmri(
        dim=[64, 64, 32],           # Spatial dimensions
        n_timepoints=100,           # Number of volumes
        tr=2.0,                     # Repetition time in seconds
        noise_sd=1.0,               # Noise standard deviation
        signal_regions=None,        # Optional: list of ROIs with signal
        signal_strength=3.0,        # Signal amplitude in activated regions
        temporal_autocorr=0.3,      # AR(1) temporal autocorrelation
        spatial_smoothness=2.0,     # FWHM of spatial smoothness
        seed=42                     # Random seed for reproducibility
    )

    print(sim_vec.shape)  # (64, 64, 32, 100)
    neuroim.write_vec(sim_vec, "simulated_fmri.nii.gz")

Simulating Task-Based Activation
---------------------------------

Create synthetic data with task-evoked responses:

.. code-block:: python

    # Define activation regions
    space = neuroim.NeuroSpace(dim=[64, 64, 32], spacing=[3, 3, 3])

    # Create ROIs where signal will be present
    roi1 = neuroim.spherical_roi(
        space,
        center=[32, 32, 16],
        radius=8.0
    )

    roi2 = neuroim.spherical_roi(
        space,
        center=[45, 30, 16],
        radius=6.0
    )

    # Create task design matrix (block design)
    n_timepoints = 200
    design = np.zeros(n_timepoints)

    # Blocks: on for 10 TRs, off for 20 TRs
    for start in range(0, n_timepoints, 30):
        design[start:start+10] = 1

    # Convolve with HRF (simplified double-gamma)
    hrf = neuroim.double_gamma_hrf(tr=2.0, duration=32)
    signal = np.convolve(design, hrf)[:n_timepoints]

    # Simulate data with activation in ROIs
    sim_vec = neuroim.simulate_fmri(
        dim=[64, 64, 32],
        n_timepoints=n_timepoints,
        tr=2.0,
        noise_sd=1.0,
        signal_regions=[roi1, roi2],
        signal_pattern=signal,
        signal_strength=2.5,
        seed=42
    )

    neuroim.write_vec(sim_vec, "task_fmri_sim.nii.gz")

Creating Confound Regressors
-----------------------------

Generate nuisance regressors for denoising:

.. code-block:: python

    # Create motion confounds (6 parameters: 3 translation, 3 rotation)
    motion = neuroim.prepare_confounds(
        n_timepoints=200,
        confound_type='motion',
        motion_amplitude=2.0,  # Maximum displacement in mm
        seed=42
    )

    print(motion.shape)  # (200, 6)

    # Create polynomial drift regressors
    drift = neuroim.prepare_confounds(
        n_timepoints=200,
        confound_type='polynomial',
        poly_degree=3
    )

    print(drift.shape)  # (200, 4) - intercept + 3 polynomial terms

    # Create discrete cosine basis for high-pass filtering
    cosine = neuroim.prepare_confounds(
        n_timepoints=200,
        confound_type='cosine',
        high_pass_cutoff=0.01,  # Hz
        tr=2.0
    )

    # Combine all confounds
    all_confounds = np.hstack([motion, drift, cosine])
    print(all_confounds.shape)

Temporal Weighting
------------------

Create temporal weights for weighted regression:

.. code-block:: python

    # Create exponential decay weights (weight recent timepoints more)
    weights = neuroim.make_time_weights(
        n_timepoints=200,
        weight_type='exponential',
        decay_rate=0.95
    )

    # Create Gaussian weights centered on a specific timepoint
    gaussian_weights = neuroim.make_time_weights(
        n_timepoints=200,
        weight_type='gaussian',
        center=100,
        sigma=20
    )

    # Create block weights (weight specific epochs)
    block_weights = neuroim.make_time_weights(
        n_timepoints=200,
        weight_type='block',
        blocks=[(20, 40), (80, 100), (140, 160)]  # Start, end tuples
    )

    # Use weights in analysis
    # weighted_data = data * weights[:, np.newaxis]

Simulating Resting-State Data
------------------------------

Create synthetic resting-state fMRI with realistic properties:

.. code-block:: python

    # Simulate resting-state with spatial and temporal structure
    rest_vec = neuroim.simulate_fmri(
        dim=[64, 64, 32],
        n_timepoints=300,
        tr=2.0,
        noise_sd=1.0,
        temporal_autocorr=0.4,      # Higher autocorrelation for resting
        spatial_smoothness=6.0,     # Typical FWHM for resting-state
        physiological_noise=True,   # Add cardiac/respiratory noise
        cardiac_freq=1.2,           # Hz
        respiratory_freq=0.3,       # Hz
        seed=42
    )

    neuroim.write_vec(rest_vec, "resting_state_sim.nii.gz")

Power Analysis
--------------

Use simulations to estimate required sample sizes:

.. code-block:: python

    from scipy import stats

    def run_power_simulation(n_subjects, effect_size, n_sims=100):
        """
        Simulate power for detecting activation.

        Returns proportion of significant results.
        """
        significant = 0

        for _ in range(n_sims):
            # Simulate data for each subject
            group_data = []

            for subj in range(n_subjects):
                vec = neuroim.simulate_fmri(
                    dim=[20, 20, 10],
                    n_timepoints=100,
                    noise_sd=1.0,
                    signal_strength=effect_size,
                    seed=None
                )
                group_data.append(vec.data[10, 10, 5, :].mean())

            # Test against zero
            t_stat, p_val = stats.ttest_1samp(group_data, 0)

            if p_val < 0.05:
                significant += 1

        return significant / n_sims

    # Estimate power for different sample sizes
    for n in [10, 20, 30, 40]:
        power = run_power_simulation(n, effect_size=0.5, n_sims=100)
        print(f"N={n}: Power={power:.2f}")
