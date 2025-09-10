import numpy as np
import h5py
from tqdm.auto import tqdm
from xrayscatteringtools.epicsArch import EpicsArchive
from scipy.interpolate import interp1d
import yaml

def combineRuns(runNumbers, folders, keys_to_combine, keys_to_sum, keys_to_check, verbose=False, archImport=False):
    """
    Combine data from multiple experimental runs into a single consolidated dataset.

    This function loads data from HDF5 files corresponding to each run, concatenates or sums
    selected keys, checks consistency of other keys, and optionally fills missing EPICS 
    gas cell pressure data from an archive. Each run's data can reside in a separate folder 
    or in a single common folder.

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
        If True, prints detailed information during data loading (default: False).
    archImport : bool, optional
        If True, enables special handling of missing gas cell pressure data from older
        archive files (default: False).

    Returns
    -------
    data_combined : dict
        Dictionary containing the combined data from all runs. Keys include:
        - Concatenated keys from `keys_to_combine`
        - Summed keys from `keys_to_sum`
        - Checked keys from `keys_to_check`
        - `'run_indicator'`: an array indicating which run each data point belongs to
        - `'epicsUser/gasCell_pressure'`: filled either from files or from the EPICS archive
          if `archImport` is True and the key was missing.

    Raises
    ------
    TypeError
        If `folders` is not a string, bytes, list, or tuple.
    ValueError
        If multiple folders are provided but the number of folders does not match
        the number of run numbers.
    """
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
        
    data_array = []
    for i,runNumber in enumerate(tqdm(runNumbers,desc="Loading Runs")):
        data = {}
        experiment = folders[i].split('/')[6]
        filename = f'{folders[i]}{experiment}_Run{runNumToString(runNumber)}.h5'
        print('Loading: ' + filename)
        with h5py.File(filename,'r') as f:
            get_leaves(f,data,verbose=verbose)
            data_array.append(data)
    data_combined = {}
    for key in keys_to_combine:
        # Special routine for loading the gas cell pressure if it was not saved. Likely a better way to do this... should talk to silke
        epicsLoad = False # Default flag value
        if (key == 'epicsUser/gasCell_pressure') & (archImport):
            try:
                arr = np.squeeze(data_array[0][key])
                for data in data_array[1:]:
                    arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
                data_combined[key] = arr
            except:
                epicsLoad = True # Set flag if we can't load from the files
        else: # All other keys load normally
            arr = np.squeeze(data_array[0][key])
            for data in data_array[1:]:
                arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
            data_combined[key] = arr
    run_indicator = np.array([])
    for i,runNumber in enumerate(runNumbers):
        run_indicator = np.concatenate((run_indicator,runNumber*np.ones_like(data_array[i]['lightStatus/xray'])))
    data_combined['run_indicator'] = run_indicator
    for key in keys_to_sum:
        arr = np.zeros_like(data_array[0][key])
        for data in data_array:
            arr += data[key]
        data_combined[key] = arr
    for key in keys_to_check:
        arr = data_array[0][key]
        for i,data in enumerate(data_array):
            if not np.array_equal(data[key],arr):
                print(f'Problem with key {key} in run {runNumbers[i]}')
        data_combined[key] = arr
    # Now to do the special gas cell pressure loading if the flag was set
    if epicsLoad:
        archive = EpicsArchive()
        unixTime = data_combined['unixTime']
        epicsPressure = np.array([]) # Init empty array
        for i,runNumber in enumerate(runNumbers):
            # Pull out start and end times from each run
            runUnixTime = unixTime[run_indicator==runNumber]
            startTime = runUnixTime[0]
            endTime = runUnixTime[-1]
            [times,pressure] = archive.get_points(PV='CXI:MKS670:READINGGET', start=startTime, end=endTime,unit="seconds",raw=True,two_lists=True); # Make Request
            # Interpolate the data
            interp_func = interp1d(times, pressure, kind='previous', fill_value='extrapolate')
            epicsPressure = np.append(epicsPressure,interp_func(runUnixTime)) # Append the data
        # Once all the data is loaded in
        data_combined['epicsUser/gasCell_pressure'] = epicsPressure # Save to the original key.       
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

def get_leaves(f, saveto, verbose=False):
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
    list of str
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
    # Ensure run_numbers is iterable
    if isinstance(run_numbers, int):
        run_numbers = [run_numbers]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = []
    for run in run_numbers:
        for entry in config['data_paths']:
            lower, upper = entry['runs']
            if lower <= run <= upper:
                paths.append(entry['path'])
                break
        else:
            raise ValueError(f"No data path found for run number: {run}")

    return paths

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
    numstr = str(num)
    while len(numstr) < 4:
        numstr = '0' + numstr
    return numstr

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

