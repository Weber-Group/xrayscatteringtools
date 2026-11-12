import glob
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
from .epicsArch import EpicsArchive
from scipy.interpolate import interp1d
import yaml
from .utils import element_number_to_symbol
from numbers import Number

_WEKA_UNITS = {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12, 'PB': 1e15}


def tier_status(paths):
    """
    Report how much of each file is on the fast SSD tier of a Weka filesystem (e.g. /sdf/data on SLAC S3DF).

    Files that have not been read for a while are kept only in Weka's object-storage tier and read at
    roughly 40-50 MB/s per file on first access. Use :func:`prefetch_files` to bring them back to SSD.

    Parameters
    ----------
    paths : str or list of str
        File path(s) to query.

    Returns
    -------
    status : dict or None
        ``{path: (size_bytes, ssd_bytes)}`` for every path Weka reported on, or None if the ``weka``
        command is not available (not on a Weka filesystem client).
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    paths = [os.fspath(p) for p in paths]
    if shutil.which('weka') is None or len(paths) == 0:
        return None
    status = {}
    for i in range(0, len(paths), 256):  # keep the command line short
        try:
            out = subprocess.run(['weka', 'fs', 'tier', 'location', *paths[i:i + 256]],
                                 capture_output=True, text=True).stdout
        except OSError:
            return None
        for line in out.splitlines()[1:]:
            tok = line.split()
            # PATH (may contain spaces) | FILE TYPE | FILE SIZE | SSD (WRITE) | SSD (READ) | OBJECT | REMOTE
            if len(tok) < 12:
                continue
            try:
                size = float(tok[-10]) * _WEKA_UNITS[tok[-9]]
                ssd = float(tok[-8]) * _WEKA_UNITS[tok[-7]] + float(tok[-6]) * _WEKA_UNITS[tok[-5]]
            except (ValueError, KeyError):
                continue
            status[' '.join(tok[:-11])] = (size, min(ssd, size))
    return status


def prefetch_files(paths, n_parallel=8, wait=False, verbose=False):
    """
    Ask a Weka filesystem (e.g. /sdf/data on SLAC S3DF) to copy files from its object-storage tier back to SSD.

    Cold files read at roughly 40-50 MB/s per file; fetching several at once in the background makes
    subsequent reads fast. Files are fetched in the order given, ``n_parallel`` at a time. Files that are
    already on SSD return almost immediately. Does nothing if the ``weka`` command is not available.

    Equivalent shell command::

        ls *.h5 | xargs -n1 -P8 weka fs tier fetch -v

    Parameters
    ----------
    paths : str or list of str
        File path(s) to fetch.
    n_parallel : int, optional
        Number of files fetched at the same time (default: 8).
    wait : bool, optional
        If True, block until every fetch has finished. If False (default), the fetches run in background
        threads and this function returns immediately.
    verbose : bool, optional
        If True, print a line when the fetches start and a warning for any fetch that fails.

    Returns
    -------
    futures : list of concurrent.futures.Future or None
        One future per file (result: the ``weka`` return code), or None if ``weka`` is not available.
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    paths = [os.fspath(p) for p in paths]
    if shutil.which('weka') is None or len(paths) == 0:
        return None

    def _fetch(path):
        try:
            res = subprocess.run(['weka', 'fs', 'tier', 'fetch', path], capture_output=True, text=True)
        except OSError as e:
            if verbose:
                print(f"weka fs tier fetch failed for {path}: {e}")
            return -1
        if res.returncode != 0 and verbose:
            print(f"weka fs tier fetch failed for {path}: {res.stderr.strip()}")
        return res.returncode

    if verbose:
        print(f"Prefetching {len(paths)} file(s) to SSD, {n_parallel} at a time")
    executor = ThreadPoolExecutor(max_workers=max(1, int(n_parallel)))
    futures = [executor.submit(_fetch, p) for p in paths]
    executor.shutdown(wait=wait)  # with wait=False the queued fetches keep running in the background
    return futures


def _fmt_bytes(n):
    """Format a byte count with decimal units, matching what ``weka`` reports."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1000 or unit == 'TB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.2f} {unit}"
        n /= 1000


def _prefetch_state(future, path):
    """One-line description of where a single file's prefetch stands (used by combineRuns(verbose=True))."""
    if future.cancelled():
        return "prefetch: cancelled"
    if future.done():
        if future.exception() is not None:
            return f"prefetch: failed ({future.exception()})"
        rc = future.result()
        return "prefetch: done, file is on SSD" if rc == 0 else f"prefetch: failed (weka exit code {rc}), reading from object storage"
    st = (tier_status(path) or {}).get(os.fspath(path))
    if st and st[1] >= st[0]:
        return "prefetch: file is already on SSD"
    on_ssd = f", {_fmt_bytes(st[1])} of {_fmt_bytes(st[0])} on SSD so far" if st else ""
    return f"prefetch: {'in progress' if future.running() else 'still queued'}{on_ssd}"


def combineRuns(runNumbers, folders, keys_to_combine, keys_to_sum, keys_to_check, verbose=False, archPVs=None,
                prefetch=True, prefetch_parallel=8):
    """
    Combine data from multiple experimental runs into a single consolidated dataset.

    This function loads data from HDF5 files corresponding to each run, concatenates or sums
    selected keys, checks consistency of other keys, and optionally imports EPICS PV data
    from the archive. Each run's data can reside in a separate folder or in a single common
    folder.

    Parameters
    ----------
    runNumbers : int or list of int
        Run number(s) to load and combine. Can be a single integer or a list of integers.
    folders : str, bytes, list, or tuple
        Path(s) to the folder(s) containing the data files. If a single string is provided,
        it is repeated for all run numbers. If multiple folders are provided, the number of
        folders must match the number of run numbers.
    keys_to_combine : list of str
        Keys in the data files whose arrays should be concatenated along the first axis.
    keys_to_sum : list of str
        Keys in the data files whose arrays should be summed element-wise across runs.
    keys_to_check : list of str
        Keys for which consistency across runs should be verified. If any discrepancies
        are found, a warning is printed.
    verbose : bool, optional
        If True, prints detailed information during data loading (default: False): the keys in each
        file, how much of the run files was already on SSD before prefetching, and the prefetch status
        of each file as it is loaded.
    archPVs : str or list of str, optional
        EPICS PV name(s) to fetch from the EPICS archive and interpolate onto the
        run timestamps. Each PV is stored in `data_combined` using the PV name as
        the key. For this to work, the 'unixTime' key must be present in each run's data.
        Note: this will only work on machines with access to the EPICS archive. (default: None).
    prefetch : bool, optional
        If True (default), first ask the Weka filesystem to bring every run file from its object-storage
        tier back to SSD, in the background and in load order (see :func:`prefetch_files`). This makes runs
        that have not been read for a while load many times faster. A one-line notice is printed when the
        prefetch starts. Silently skipped when the ``weka`` command is not available (e.g. off the cluster).
        Fetches still queued when loading finishes (or fails) are cancelled.
    prefetch_parallel : int, optional
        Number of files prefetched at the same time (default: 8).

    Returns
    -------
    data_combined : dict
        Dictionary containing the combined data from all runs. Keys include:
        - Concatenated keys from `keys_to_combine`
        - Summed keys from `keys_to_sum`
        - Checked keys from `keys_to_check`
        - `'run_indicator'`: an array indicating which run each data point belongs to
        - One key per PV in `archPVs`, containing the interpolated archive data.

    Raises
    ------
    TypeError
        If `folders` is not a string, bytes, list, or tuple.
    ValueError
        If multiple folders are provided but the number of folders does not match
        the number of run numbers.

    Notes
    -----
    If unixTime is not present in the .h5 files, add the following code into the event loop of the producer script to save the unix timestamps.
    ```python
    unixTimeDict = {}
    evtid = evt.get(psana.EventId)
    tsec, tnsec = evtid.time()
    unixTimeDict['unixTime'] = tsec+tnsec/1e9
    small_data.event(unixTimeDict)
    ```
    """
    from tqdm.auto import tqdm # Lazy
    # Ensure runNumbers is a list
    if not isinstance(runNumbers, (list, tuple)):
        runNumbers = [runNumbers]

    # If folders is a string or bytes, make it a list of one element
    if isinstance(folders, (str, bytes)):
        folders = [folders]
    # If folders is a tuple, convert to list
    elif isinstance(folders, tuple):
        folders = list(folders)
    # If folders is already a list, keep as is
    elif not isinstance(folders, list):
        raise TypeError(f"'folders' must be a string, bytes, list, or tuple. Got {type(folders)}")

    if len(folders) > 1:
        if len(folders) != len(runNumbers):
            raise ValueError(
                f"If 'folders' has more than one element, its length must match 'runNumbers'. "
                f"Got len(runNumbers)={len(runNumbers)} and len(folders)={len(folders)}."
            )
    else:
        # Repeat the single folders to match the length of runNumbers
        folders = folders * len(runNumbers)
        
    # Build the set of keys we actually need to load from each file
    needed_keys = set(keys_to_combine) | set(keys_to_sum) | set(keys_to_check)
    needed_keys.add('lightStatus/xray')  # Always needed for run_indicator
    if archPVs is not None:
        needed_keys.add('unixTime')  # Needed for EPICS PV interpolation

    # Resolve every run's file up front (so a missing run fails before anything is loaded)
    filenames = []
    for i, runNumber in enumerate(runNumbers):
        run_suffix = f'_Run{runNumToString(runNumber)}.h5'
        matches = sorted(glob.glob(os.path.join(folders[i], f'*{run_suffix}')))
        if len(matches) == 0:
            raise FileNotFoundError(f"No files found in {folders[i]} matching pattern '*{run_suffix}'")
        elif len(matches) > 1:
            raise FileExistsError(f"Multiple files found in {folders[i]} matching pattern '*{run_suffix}': {matches}")
        filenames.append(matches[0])

    # Bring cold files back from the object-storage tier while the first ones are being read
    prefetch_futures = None
    if prefetch:
        before = tier_status(filenames) if verbose else None  # snapshot before fetching starts
        prefetch_futures = prefetch_files(filenames, n_parallel=prefetch_parallel, wait=False)
        if prefetch_futures is None:
            if verbose:
                print("Prefetch skipped: 'weka' command not available (not on a Weka filesystem client)")
        else:
            print(f"Prefetching {len(filenames)} run file(s) to SSD in the background "
                  f"({max(1, int(prefetch_parallel))} at a time)")
            if verbose and before:
                total = sum(s[0] for s in before.values())
                ssd = sum(s[1] for s in before.values())
                n_hot = sum(s[1] >= s[0] for s in before.values())
                print(f"  Already on SSD: {_fmt_bytes(ssd)} of {_fmt_bytes(total)} "
                      f"({n_hot}/{len(filenames)} files fully on SSD)")

    data_array = []
    try:
        for i, runNumber in enumerate(tqdm(runNumbers, desc="Loading Runs")):
            data = {}
            filename = filenames[i]
            print('Loading: ' + filename)
            if verbose and prefetch_futures is not None:
                print('  ' + _prefetch_state(prefetch_futures[i], filename))
            with h5py.File(filename, 'r') as f:
                # Print all keys and shapes without loading data
                if verbose:
                    def _print_leaf(name):
                        item = f[name]
                        if isinstance(item, h5py.Dataset):
                            print(f"  {name}  {item.shape}  {item.dtype}")
                    f.visit(_print_leaf)
                # Only load the keys we need
                for key in needed_keys:
                    if key in f:
                        data[key] = f[key][()]
                        if verbose:
                            print(f"  [loaded] {key}")
                data_array.append(data)
    finally:
        # Don't keep fetching files nobody is going to read (e.g. after an error or Ctrl-C)
        if prefetch_futures is not None:
            for fut in prefetch_futures:
                fut.cancel()

    data_combined = {}
    for key in tqdm(keys_to_combine, desc="Combining Data"):
        chunks = [np.squeeze(d[key]) for d in data_array]
        data_combined[key] = np.concatenate(chunks, axis=0)

    run_indicator = np.concatenate([
        runNumber * np.ones_like(data_array[i]['lightStatus/xray'])
        for i, runNumber in enumerate(runNumbers)
    ])
    data_combined['run_indicator'] = run_indicator

    for key in tqdm(keys_to_sum, desc="Summing Data"):
        arr = np.zeros_like(data_array[0][key])
        for data in data_array:
            arr += data[key]
        data_combined[key] = arr

    for key in tqdm(keys_to_check, desc="Checking Data"):
        arr = data_array[0][key]
        for i, data in enumerate(data_array):
            if not np.array_equal(data[key], arr):
                print(f'Problem with key {key} in run {runNumbers[i]}')
        data_combined[key] = arr
    # Import EPICS PV data from the archive if requested
    if archPVs is not None:
        if isinstance(archPVs, str):
            archPVs = [archPVs]
        archive = EpicsArchive()
        for i, data in enumerate(data_array):
            if 'unixTime' not in data:
                raise KeyError(f"'unixTime' not found in run {runNumbers[i]}; it is required for archPVs.")
        unixTime = np.concatenate([np.squeeze(d['unixTime']) for d in data_array])
        for pv in tqdm(archPVs, desc="Loading EPICS PVs"):
            pv_chunks = []
            for i, runNumber in enumerate(runNumbers):
                runUnixTime = unixTime[run_indicator == runNumber]
                startTime = runUnixTime[0]
                endTime = runUnixTime[-1]
                [times, values] = archive.get_points(
                    PV=pv, start=startTime, end=endTime,
                    unit="seconds", raw=True, two_lists=True
                )
                interp_func = interp1d(times, values, kind='previous', fill_value='extrapolate')
                pv_chunks.append(interp_func(runUnixTime))
            data_combined[pv] = np.concatenate(pv_chunks)
    print('Loaded Data')
    return data_combined

def get_tree(f):
    """List the full tree of the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        The HDF5 file object to traverse.

    Returns
    -------
    None
        Prints the structure of the HDF5 file.
    """
    def printname(name):
        print(name, type(f[name]))
    f.visit(printname)
    
def is_leaf(dataset):
    """Check if an HDF5 node is a dataset (leaf node).

    Parameters
    ----------
    dataset : h5py.Dataset or h5py.Group
        The HDF5 node to check.

    Returns
    -------
    bool
        True if the node is a dataset, False otherwise.
    """
    return isinstance(dataset, h5py.Dataset)

def get_leaves(f, saveto=None, verbose=False):
    """Retrieve all leaf datasets from an HDF5 file and save them to a dictionary.

    Parameters
    ----------
    f : h5py.File
        The HDF5 file object to traverse.
    saveto : dict
        Dictionary to store the retrieved datasets.
    verbose : bool, optional
        If True, print detailed information about each dataset (default: False).

    Returns
    -------
    None
        The datasets are stored in the provided dictionary.
    """
    def return_leaf(name):
        if is_leaf(f[name]):
            if verbose:
                print(name, f[name][()].shape)
            if saveto is not None:
                saveto[name] = f[name][()]
    f.visit(return_leaf)

def get_data_paths(run_numbers, config_path='config.yaml'):
    """
    Retrieve data directories for a list of run numbers.

    Reads a YAML configuration file specifying run ranges and associated
    data paths, returning the paths corresponding to the provided run numbers.

    Parameters
    ----------
    run_numbers : int or iterable of int
        Run number(s) for which the data paths are requested.
    config_path : str, optional
        Path to the YAML configuration file (default is 'config.yaml').

    Returns
    -------
    str or list of str
        Data directory paths corresponding to each run number.

    Raises
    ------
    ValueError
        If any run number does not have a corresponding data path.
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML configuration file.

    Examples
    --------
    >>> get_data_paths([5, 15, 25], 'config.yaml')
    ['/sdf/data/lcls/ds/cxi/cxil', '/sdf/data/lcls/ds/cxi/cxil2', '/sdf/data/lcls/ds/cxi/cxil3']
    """
    return get_config_for_runs(run_numbers,'data_paths','path',config_path=config_path)

def get_config(key, config_path='config.yaml'):
    """
    Retrieve configuration values for a specific key from a YAML configuration file.

    Parameters
    ----------
    key : str
        The configuration key to retrieve values for.
    config_path : str, optional
        Path to the YAML configuration file (default is 'config.yaml').

    Returns
    -------
    list
        The configuration values corresponding to the specified key.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config[key]

def get_config_for_runs(run_numbers,key,subkey,config_path='config.yaml'):
    """
    Retrieve configuration values for a specific key based on run number ranges.

    Reads a YAML configuration file specifying run ranges and associated
    configuration values, returning the values corresponding to the provided key.

    Parameters
    ----------
    run_numbers : int or iterable of int
        Run number(s) for which the configuration values are requested.
    key : str
        The configuration key to retrieve values for.
    subkey : str
        The subkey within the configuration entries to extract the value from.
    config_path : str, optional
        Path to the YAML configuration file (default is 'config.yaml').

    Returns
    -------
    variable or list
        The configuration values corresponding to the specified key.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML configuration file.
    """
    # Ensure run_numbers is iterable
    if isinstance(run_numbers, Number):
        run_numbers = [run_numbers]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    values = []
    for run in run_numbers:
        for entry in config[key]:
            lower, upper = entry['runs']
            if lower <= run <= upper:
                values.append(entry[subkey])
                break
        else:
            raise ValueError(f"No {key}/{subkey} value found for run number: {run}")
    # Return scalar if only one value
    if len(values) == 1:
        return values[0]

    return values

def runNumToString(num):
    """Convert a run number to a zero-padded string of length 4.

    Parameters
    ----------
    num : int
        The run number to convert.

    Returns
    -------
    numstr : str
        The zero-padded string representation of the run number.
    """
    return str(num).zfill(4)

def read_xyz(filename):
    """
    Read a molecular structure from an XYZ file.

    Parameters
    ----------
    filename : str
        Path to the XYZ file to be read.

    Returns
    -------
    num_atoms : int
        Number of atoms in the XYZ file.
    comment : str
        Comment line from the XYZ file (usually contains metadata or description).
    atoms : list of str
        List of atomic symbols (e.g., 'C', 'H', 'O') for each atom.
    coords : list of tuple of float
        Cartesian coordinates of each atom as (x, y, z) in the same units as the file.

    Notes
    -----
    The XYZ file format is expected to have:
    1. First line: number of atoms.
    2. Second line: comment or description.
    3. Following lines: atomic symbol and x, y, z coordinates for each atom.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    atoms = []
    coords = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        coords.append((x, y, z))
    
    return num_atoms, comment, atoms, coords

def write_xyz(filename, comment, atoms, coords):
    """
    Save atomic coordinates to an XYZ file.

    Parameters
    ----------
    filename : str
        The name of the output XYZ file.
    comment : str
        A comment line to include in the XYZ file.
    atoms : list
        List of atomic symbols or numbers.
    coords : np.ndarray
        Nx3 array of atomic coordinates (x, y, z).
    """
    # Convert atomic numbers to symbols if necessary
    atoms = [element_number_to_symbol(atom) if isinstance(atom, int) else atom for atom in atoms]
    # Write the data to the file
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

def read_mol(filename):
    """
    Read a molecular structure from a MOL file (MDL Molfile format).

    Parameters
    ----------
    filename : str
        Path to the MOL file to be read.

    Returns
    -------
    molecule_name : str
        Name of the molecule (first line of the file).
    program_info : str
        Program/user information (second line).
    comment : str
        Comment line (third line).
    num_atoms : int
        Number of atoms in the molecule.
    num_bonds : int
        Number of bonds in the molecule.
    atoms : list of str
        List of atomic symbols for each atom.
    coords : list of tuple of float
        Cartesian coordinates of each atom as (x, y, z) in Angstroms.
    bonds : list of tuple of int
        List of bonds as (atom1_index, atom2_index, bond_type).
        Atom indices are 1-based as in the MOL file.
        Bond types: 1=single, 2=double, 3=triple, 4=aromatic.
    properties : dict
        Additional properties from the atom and bond blocks.
        - 'atom_properties': list of dicts with mass_diff, charge, stereo_parity, etc.
        - 'bond_properties': list of dicts with stereo, topology, etc.

    Notes
    -----
    The MOL file format (V2000) is expected to have:
    1. Line 1: Molecule name
    2. Line 2: Program/user info and timestamp
    3. Line 3: Comment
    4. Line 4: Counts line (atoms, bonds, etc.)
    5. Atom block: one line per atom with coordinates and properties
    6. Bond block: one line per bond
    7. Properties block (optional)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Header block (first 3 lines)
    molecule_name = lines[0].strip()
    program_info = lines[1].strip()
    comment = lines[2].strip()
    
    # Counts line (line 4)
    counts_line = lines[3]
    num_atoms = int(counts_line[0:3].strip())
    num_bonds = int(counts_line[3:6].strip())
    
    # Atom block
    atoms = []
    coords = []
    atom_properties = []
    
    for i in range(4, 4 + num_atoms):
        line = lines[i]
        x = float(line[0:10].strip())
        y = float(line[10:20].strip())
        z = float(line[20:30].strip())
        element = line[31:34].strip()
        
        # Additional atom properties
        mass_diff = int(line[34:36].strip()) if len(line) > 34 and line[34:36].strip() else 0
        charge = int(line[36:39].strip()) if len(line) > 36 and line[36:39].strip() else 0
        
        atoms.append(element)
        coords.append((x, y, z))
        atom_properties.append({
            'mass_diff': mass_diff,
            'charge': charge
        })
    
    # Bond block
    bonds = []
    bond_properties = []
    
    bond_start = 4 + num_atoms
    for i in range(bond_start, bond_start + num_bonds):
        line = lines[i]
        atom1 = int(line[0:3].strip())
        atom2 = int(line[3:6].strip())
        bond_type = int(line[6:9].strip())
        
        # Additional bond properties
        stereo = int(line[9:12].strip()) if len(line) > 9 and line[9:12].strip() else 0
        
        bonds.append((atom1, atom2, bond_type))
        bond_properties.append({
            'stereo': stereo
        })
    
    properties = {
        'atom_properties': atom_properties,
        'bond_properties': bond_properties
    }
    
    return molecule_name, program_info, comment, num_atoms, num_bonds, atoms, coords, bonds, properties