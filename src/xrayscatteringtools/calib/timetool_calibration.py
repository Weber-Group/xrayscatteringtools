import numpy as np
import math
from ruamel.yaml import YAML

def fast_erf_fit(array, min_val=0.1, max_val=0.9):
    """
    Estimate the center and characteristics of an error-function-like curve.
    
    Normalizes the input data to the range 0–1, identifies the positions where 
    the normalized curve crosses specified threshold values, and calculates 
    various properties of the transition region.

    Parameters
    ----------
    array : array_like
        1D array containing the data to fit. Should resemble an error function 
        or step-like transition.
    min_val : float, optional
        Lower threshold value (fraction of normalized range) for determining 
        the transition region. Default is 0.1.
    max_val : float, optional
        Upper threshold value (fraction of normalized range) for determining 
        the transition region. Default is 0.9.

    Returns
    -------
    range_val : int
        Width of the transition region in array indices (max_pos - min_pos).
    cent_pos : int
        Index of the center position of the transition region.
    cent_amp : float
        Normalized amplitude at the center position.
    norm_data : ndarray
        Normalized version of the input array (scaled to 0–1).
    slope_val : float
        Estimated slope of the transition (rise over run).

    Notes
    -----
    If the input array is flat (constant values), or if the threshold crossings 
    cannot be found, the function returns zeros for all values except norm_data, 
    which will be a zero array.
    """
    array = np.asarray(array)
    length = len(array)

    # Normalize
    lo, hi = np.nanmin(array), np.nanmax(array)
    if hi == lo:
        # Flat array → return safe defaults
        return 0, 0, 0, np.zeros(length), 0

    norm = (array - lo) / (hi - lo)

    # Indices where norm < min_val and norm > max_val
    low_idx = np.where(norm < min_val)[0]
    high_idx = np.where(norm > max_val)[0]

    if len(low_idx) == 0 or len(high_idx) == 0:
        # Thresholds not found
        return 0, 0, 0, np.zeros(length), 0

    min_pos  = low_idx.max()
    max_pos  = high_idx.min()

    if min_pos >= max_pos:
        # Invalid ordering
        return 0, 0, 0, np.zeros(length), 0

    # Core quantities
    range_val = max_pos - min_pos
    cent_pos  = (min_pos + max_pos) // 2
    cent_amp  = norm[cent_pos]
    slope_val = (norm[max_pos] - norm[min_pos]) / range_val

    return range_val, cent_pos, cent_amp, norm, slope_val

def add_calibration_to_yaml(run_range, slope, intercept, file_path = 'config.yaml' ,key_name='tt_calibration'):
    """Append timetool calibration data to a YAML file, preserving comments.

    The function loads an existing YAML file using :class:`ruamel.yaml.YAML`
    (preserving comments and quoting), appends a new calibration entry under
    ``key_name`` and writes the updated tree back to disk. The calibration
    entry contains a list of runs (the supplied ``run_range``), a slope and
    an intercept.

    Parameters
    ----------
    run_range : sequence
        Sequence of run identifiers. Elements may be numbers or the string
        ``'.inf'`` (case-insensitive) to indicate an open-ended range; the
        latter will be converted to ``float('inf')`` before being written.
    slope : float
        Slope value for the timetool calibration entry.
    intercept : float
        Intercept value for the timetool calibration entry.
    file_path : str, optional
        Path to the YAML file to modify. Default is ``'config.yaml'``.
    key_name : str, optional
        Top-level key under which calibration entries are stored. If the key
        does not exist it will be created. Default is ``'tt_calibration'``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified ``file_path`` does not exist when attempting to
        open it for reading. Note: the implementation currently catches
        ``FileNotFoundError`` and prints an error message instead of
        re-raising; callers should be aware of this behavior.

    Notes
    -----
    - This function uses :mod:`ruamel.yaml` to preserve comments and
      formatting in the existing YAML file. The ``runs`` sequence is forced
      to flow style (e.g. ``[30, .inf]``) when written.
    - The function modifies the file in-place by overwriting it.

    Examples
    --------
    >>> add_calibration_to_yaml([30, '.inf'], 0.123, 4.56, file_path='config.yaml')

    """
    yaml = YAML()
    yaml.preserve_quotes = True 
    
    # 1. Load the existing YAML data
    try:
        with open(file_path, 'r') as f:
            data = yaml.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
        
    # 2. Ensure the target key exists; if not, initialize it as an empty list
    if key_name not in data or data[key_name] is None:
        data[key_name] = []
        
    # 3. Process the run range to handle '.inf' correctly
    processed_runs = []
    for r in run_range:
        # If the user passes the string '.inf', convert it to a float infinity.
        # ruamel.yaml will automatically write float('inf') as .inf in the file.
        if isinstance(r, str) and r.lower() in ['.inf', 'inf', '+inf']:
            processed_runs.append(float('inf'))
        else:
            processed_runs.append(r)
            
    # Optional: Force the 'runs' list to print inline (e.g., [30, .inf]) instead of on new lines
    runs_seq = yaml.seq(processed_runs)
    runs_seq.fa.set_flow_style()
            
    # 4. Construct the new entry
    new_entry = {
        'runs': runs_seq,
        'slope': float(slope),
        'intercept': float(intercept)
    }
    
    # 5. Append the new entry to the calibration list
    data[key_name].append(new_entry)
    
    # 6. Write the updated data back to the file
    with open(file_path, 'w') as f:
        yaml.dump(data, f)
        
    print(f"Successfully appended calibration to '{key_name}' in {file_path}.")

def apply_timetool_correction(delays, edge_positions, slopes, intercepts, run_indicator=None):
    """
    Apply timetool corrections, optionally using per-run calibration parameters.

    Parameters
    ----------
    delays : array_like
        Per-shot delay values to correct.
    edge_positions : array_like
        Per-shot measured edge positions (pixels) used to compute the correction.
    slopes : scalar, sequence or ndarray
        Slope(s) for the correction (seconds per pixel). If a single value
        is provided, it will be applied to all shots. If multiple values
        are provided, their length must match the number of unique runs
        in `run_indicator`.
    intercepts : scalar, sequence or ndarray
        Intercept(s) for the correction (seconds). If a single value is
        provided, it will be applied to all shots. If multiple values
        are provided, their length must match the number of unique runs
        in `run_indicator`.
    run_indicator : array_like, optional
        Per-shot run identifier. If provided, it must have the same length
        as `delays` and `edge_positions`. The unique values in `run_indicator`
        determine which slope and intercept to use for each shot.

    Returns
    -------
    corrected_delays : ndarray
        Array of corrected delays, same shape as `delays`.
    """
    delays = np.asarray(delays)
    edge_positions = np.asarray(edge_positions)

    if run_indicator is None:
        # Single calibration case
        if not np.isscalar(slopes) or not np.isscalar(intercepts):
            raise ValueError("When run_indicator is not provided, slopes and intercepts must be scalars.")

        timetool_correction = edge_positions * slopes + intercepts
        corrected_delays = delays + timetool_correction

    else:
        # Multi-calibration case
        run_indicator = np.asarray(run_indicator)
        unique_runs = np.unique(run_indicator)
        num_unique_runs = unique_runs.size

        slopes = np.asarray(slopes)
        intercepts = np.asarray(intercepts)

        # Validate slopes and intercepts
        if slopes.ndim == 0:
            slopes = slopes.reshape(1)
        if intercepts.ndim == 0:
            intercepts = intercepts.reshape(1)

        if slopes.size == 1:
            slopes = np.full(num_unique_runs, slopes.item())
        if intercepts.size == 1:
            intercepts = np.full(num_unique_runs, intercepts.item())

        if slopes.size != num_unique_runs or intercepts.size != num_unique_runs:
            raise ValueError("Number of slopes/intercepts must be 1 or match the number of unique runs in run_indicator.")

        if run_indicator.shape != delays.shape or run_indicator.shape != edge_positions.shape:
            raise ValueError("run_indicator, delays, and edge_positions must have the same shape.")

        # Prepare output array
        corrected_delays = np.empty_like(delays, dtype=float)

        # Apply the corresponding calibration to each subset of shots
        for i, run in enumerate(unique_runs):
            mask = run_indicator == run
            if not np.any(mask):
                continue
            s = float(slopes[i])
            it = float(intercepts[i])
            timetool_correction = edge_positions[mask] * s + it
            corrected_delays[mask] = delays[mask] + timetool_correction

    return corrected_delays