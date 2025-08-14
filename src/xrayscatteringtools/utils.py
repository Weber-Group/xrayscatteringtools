import re
from IPython import get_ipython

def enable_underscore_cleanup():
    """Registers a post-cell hook to delete user-defined _ variables after each cell."""
    ipython = get_ipython()
    user_ns = ipython.user_ns  # This gives you access to the Jupyter notebook namespace

    def clean_user_underscore_vars(*args, **kwargs):
        def is_user_defined_underscore(var):
            return (
                var.startswith('_')
                and not re.match(r'^_i\d*$|^_\d*$|^_ih$|^_oh$|^_ii*$|^_iii$|^_dh$|^_$', var)
                and not var.startswith('__')
            )

        for var in list(user_ns):
            if is_user_defined_underscore(var):
                del user_ns[var]

    ipython.events.register('post_run_cell', clean_user_underscore_vars)

def keV2Angstroms(keV):
    """Convert energy in keV to wavelength in Angstroms."""
    return 12.39841984/keV

def Angstroms2keV(angstroms):
    """Convert wavelength in Angstroms to energy in keV."""
    return 12.39841984/angstroms