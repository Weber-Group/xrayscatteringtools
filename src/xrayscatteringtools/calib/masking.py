from ..utils import OCTAL, azimuthalBinning, keV2Angstroms
# from ..io import combineRuns
from ..io import combine_cubedata
import numpy as np
import matplotlib.pyplot as plt
# from ..plotting import plot_j4m
# import psana
from tqdm.auto import tqdm
import os
import h5py
from scipy import ndimage as ndi
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
# 

class mask_maker(object):
    """
    A class for creating and managing masks for X-ray scattering data.

    Attributes
    ----------
    verbose : bool
        Whether to print verbose output during processing.
    experiment : str
        The experiment identifier.
    data_path : str
        Path to the data directory.
    dark_run_number : int
        Run number for the dark data.
    background_run_number : int
        Run number for the background data.
    sample_run_number : int
        Run number for the sample data.
    dark_data : dict
        Combined data for the dark run.
    background_data : dict
        Combined data for the background run.
    sample_data : dict
        Combined data for the sample run.
    dark_mask : ndarray
        Mask for the dark data.
    background_mask : ndarray
        Mask for the background data.
    sample_mask : ndarray
        Mask for the sample data.
    cmask : ndarray
        Combined mask.
    """

    def __init__(
            self,
            experiment,
            data_path,
            dark_run_number,
            background_run_number,
            sample_run_number,
            verbose=False,
            prefix='cubeRun_',
            suffix=''
            ):
        """
        Initialize the mask_maker class.

        Parameters
        ----------
        experiment : str
            The experiment identifier.
        data_path : str
            Path to the data directory.
        dark_run_number : int
            Run number for the dark data.
        background_run_number : int
            Run number for the background data.
        sample_run_number : int
            Run number for the sample data.
        verbose : bool, optional
            Whether to print verbose output during processing (default is False).
        """
        # Setting class attrubutes
        self.verbose = verbose
        self.experiment = experiment
        self.data_path = data_path
        self.dark_run_number = dark_run_number
        self.background_run_number = background_run_number
        self.sample_run_number = sample_run_number
        self.prefix = prefix
        self.suffix = suffix

        # # Keys for processing
        # keys_to_combine = [
        #     'lightStatus/xray',
        #     'lightStatus/laser',
        #     'jungfrau4M/azav_azav',
        # ]
        # keys_to_sum = [
        #     'Sums/jungfrau4M_calib_xrayOn_thresADU1',
        #     'Sums/jungfrau4M_calib_dropped',
        # ]
        # keys_to_check = [

        # ]

        # Loading data, saving to class attributes
        # self.dark_data = combineRuns(dark_run_number, data_path, keys_to_combine, keys_to_sum, keys_to_check, verbose=verbose)
        # self.background_data = combineRuns(background_run_number, data_path, keys_to_combine, keys_to_sum, keys_to_check, verbose=verbose)
        # self.sample_data = combineRuns(sample_run_number, data_path, keys_to_combine, keys_to_sum, keys_to_check, verbose=verbose)

        ### SCALA
        _,dark_tbin_counts,dark_cubedata,_,_,_ = combine_cubedata(self.dark_run_number, self.data_path, prefix = self.prefix, suffix=self.suffix)
        self.dark_data = np.nansum(dark_cubedata,axis=0)/np.nansum(dark_tbin_counts,axis=0)
        _,bkg_tbin_counts,backround_cubedata,_,_,_ = combine_cubedata(self.background_run_number, self.data_path, prefix = self.prefix, suffix=self.suffix)
        self.background_data = np.nansum(backround_cubedata,axis=0)/np.nansum(bkg_tbin_counts,axis=0)
        _,sample_tbin_counts,sample_cubedata,_,_,_ = combine_cubedata(self.sample_run_number, self.data_path, prefix = self.prefix, suffix=self.suffix)
        self.sample_data = np.nansum(sample_cubedata,axis=0)/np.nansum(sample_tbin_counts,axis=0)
        
        # Initalize masks
        self.dark_mask = np.ones((2399,2399))
        self.background_mask = np.ones((2399,2399))
        self.sample_mask = np.ones((2399,2399))
        self.cmask = np.ones((2399,2399))
        return
    
    def process_dark(self, plotting=True, lb=None, ub=None):
        """
        Process the dark data to create a mask based on intensity thresholds.

        Parameters
        ----------
        plotting : bool, optional
            Whether to plot histograms and masks (default is True).
        lb : float, optional
            Lower bound for intensity cutoff (default is None, prompts user input).
        ub : float, optional
            Upper bound for intensity cutoff (default is None, prompts user input).

        Returns
        -------
        None
        """
        # Computing average dark image
        # dark_num_shots = np.where(self.dark_data['lightStatus/xray'].astype(bool)==False)[0].shape[0]
        # dark_avg = np.array(self.dark_data['Sums/jungfrau4M_calib_dropped'] / dark_num_shots)

        if plotting:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.hist(np.arcsinh(self.dark_data).flatten(), bins=200)
            plt.title(f'Run {self.dark_run_number} Dark Image Histogram')
            plt.yscale('log')
            plt.xlabel('arcsinh(Intensity)')
            plt.ylabel('Counts')
            plt.grid(True)

            plt.subplot(1,2,2)
            plt.hist(np.arcsinh(self.dark_data).flatten(), bins=200)
            plt.title(f'Run {self.dark_run_number} Dark Image Histogram')
            plt.yscale('linear')
            plt.xlabel('arcsinh(Intensity)')
            plt.ylabel('Counts')
            plt.grid(True)
            
            plt.show()
        # Determine recommended bounds
        recommended_bounds = np.percentile(np.arcsinh(self.dark_data).flatten(), [35, 99.9])
        if ub is None or lb is None:
            print('Determine intensity cutoffs for dark mask (Leave blank to use recommended bounds):')
            lb = float(input(f'Enter lower bound for dark mask (Recomended: {recommended_bounds[0]}): ') or recommended_bounds[0])
            ub = float(input(f'Enter upper bound for dark mask (Recomended: {recommended_bounds[1]}): ') or recommended_bounds[1])
        else:
            lb = float(lb)
            ub = float(ub)
        assert float(lb) < float(ub), "Lower bound must be less than upper bound"

        # Make the mask
        self.dark_mask = (np.arcsinh(self.dark_data) <= ub) * (np.arcsinh(self.dark_data) >= lb)

        if plotting:
            # Apply the mask for plotting
            dark_avg_masked = np.copy(self.dark_data)
            dark_avg_masked[~self.dark_mask] = np.nan

            fig2, ax2 = plt.subplots(ncols=3, figsize=(18, 8))
            # Plot dark mask
            pcm = ax2[0].imshow(self.dark_mask,vmin=0, vmax=1)
            ax2[0].set_title('Mask from Dark')
            ax2[0].set_xticks([])
            ax2[0].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[0], fraction=0.046, pad=0.04)
            # Plot arcsinh dark average
            pcm = ax2[1].imshow(np.arcsinh(self.dark_data),cmap='jet')
            ax2[1].set_title('arcsinh(Dark Average)')
            ax2[1].set_xticks([])
            ax2[1].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[1], fraction=0.046, pad=0.04)
            # Plot masked dark average
            pcm = ax2[2].imshow(dark_avg_masked,cmap='jet')
            ax2[2].set_title('Masked Dark Average')
            ax2[2].set_xticks([])
            ax2[2].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[2], fraction=0.046, pad=0.04)
            plt.show()
        # Reporting masked percentage
        print(f'Masked percentage from dark: {100*(1-np.sum(self.dark_mask)/self.dark_mask.size):.2f}%')
        return

    def process_background(self, plotting=True, lb=None, ub=None):
        """
        Process the background data to create a mask based on intensity thresholds.

        Parameters
        ----------
        plotting : bool, optional
            Whether to plot histograms and masks (default is True).
        lb : float, optional
            Lower bound for intensity cutoff (default is None, prompts user input).
        ub : float, optional
            Upper bound for intensity cutoff (default is None, prompts user input).

        Returns
        -------
        None
        """
        # Computing average background image
        background_avg_darkmask = np.copy(self.background_data)
        background_avg_darkmask[~self.dark_mask] = np.nan

        if plotting:
            # ub_plotting = np.percentile(background_avg[self.dark_mask], 99.9)

            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.hist(background_avg_darkmask.flatten(), bins=500, range=(0,np.nanmax(background_avg_darkmask)))
            plt.title(f'Run {self.background_run_number} Background Image Histogram')
            plt.xlabel('Intensity')
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.grid(True)

            plt.subplot(1,2,2)
            plt.hist(background_avg_darkmask.flatten(), bins=500, range=(0,np.nanmax(background_avg_darkmask)))
            plt.title(f'Run {self.background_run_number} Background Image Histogram')
            plt.xlabel('Intensity')
            plt.ylabel('Counts')
            plt.yscale('linear')
            plt.grid(True)
            plt.show()

        # Determine recommended bounds
        recommended_bounds = [0, np.nanpercentile(background_avg_darkmask.flatten(), 99)]
        if lb is None or ub is None:
            print('Determine intensity cutoffs for background mask (Leave blank to use recommended bounds):')
            lb = float(input(f'Enter lower bound for background mask (Recommended: {recommended_bounds[0]}): ') or recommended_bounds[0])
            ub = float(input(f'Enter upper bound for background mask (Recommended: {recommended_bounds[1]}): ') or recommended_bounds[1])
        else:
            lb = float(lb)
            ub = float(ub)
        assert lb < ub, "Lower bound must be less than upper bound"

        # Make the mask
        self.background_mask = (np.nan_to_num(background_avg_darkmask) <= ub) * (np.nan_to_num(background_avg_darkmask) >= lb)

        if plotting:
            # Apply dark mask to background average for plotting
            background_avg_backmask = np.copy(background_avg_darkmask)
            # Apply background mask
            background_avg_backmask[~(self.background_mask)] = np.nan

            fig2, ax2 = plt.subplots(ncols=3, figsize=(18, 8))
            # Plot background mask
            pcm = ax2[0].imshow(self.background_mask,vmin=0, vmax=1)
            ax2[0].set_title('Mask from Bkg')
            ax2[0].set_xticks([])
            ax2[0].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[0], fraction=0.046, pad=0.04)
            # Plot arcsinh background average with dark mask applied
            pcm = ax2[1].imshow(np.arcsinh(background_avg_darkmask),cmap='jet',vmin=0)
            ax2[1].set_title('arcsinh(Dark Masked Bkg Average)')
            ax2[1].set_xticks([])
            ax2[1].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[1], fraction=0.046, pad=0.04)
            # Plot masked background average
            pcm = ax2[2].imshow(background_avg_backmask,cmap='jet',vmin=0)
            ax2[2].set_title('Masked Bkg Average')
            ax2[2].set_xticks([])
            ax2[2].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[2], fraction=0.046, pad=0.04)
            plt.show()
        # Reporting masked percentage
        print(f'Masked percentage from background: {100*(1-np.sum(self.background_mask)/self.background_mask.size):.2f}%')
        return

    def process_sample(self, n_std=2, x0=0, y0=0, z0=90_000, tx=0, ty=0, keV=10, z_off=0, num_points = 4, points=[], plotting=True):
        """
        Process the sample data to create a mask based on intensity statistics.

        Parameters
        ----------
        n_std : int, optional
            Number of standard deviations for intensity cutoff (default is 2).
        x0 : float, optional
            X-coordinate of the beam center in microns (default is 0).
        y0 : float, optional
            Y-coordinate of the beam center in microns (default is 0).
        z0 : float, optional
            Distance from the detector to the sample in microns (default is 90000).
        tx : float, optional
            Tilt angle in the x-direction (default is 0).
        ty : float, optional
            Tilt angle in the y-direction (default is 0).
        keV : float, optional
            X-ray energy in keV (default is 10).
        z_off : float, optional
            Offset in the z-direction in microns (default is 0).
        num_pts : int, optional
            Number of points for polygon selection mask.
        points : ndarray, optional
            List of preset points. If the amount of points is less than num_points it will prompt the user to supply the rest.
        plotting : bool, optional
            Whether to plot histograms and masks (default is True).

        Returns
        -------
        None
        """
        # Init sample mask
        # self.sample_mask = np.ones_like(J4M.x)

        # Grabbing the data from the sample_data attribute, calculating average
        # sample_num_shots = np.where(self.sample_data['lightStatus/xray'].astype(bool)==True)[0].shape[0]
        # sample_avg = np.array(self.sample_data['Sums/jungfrau4M_calib_xrayOn_thresADU1'] / sample_num_shots)
        # Applying dark and background mask to sample average
        self.sample_mask = np.ones_like(self.sample_data)
        sample_avg_darkbackmask = np.copy(self.sample_data)
        sample_avg_darkbackmask[~(self.dark_mask * self.background_mask)] = np.nan

        num_points
        if num_points < 3:
            raise ValueError("At least 3 points are required to form a polygon.")
        
        _temp_mask = np.ones_like(self.sample_data, dtype=bool)  # boolean mask for display stacking
        entered_points = []
        print('Beginning polygon masking')

        
        
        for pointIdx in range(num_points):
            if pointIdx < len(points):
                entered_points.append((np.array(points)[pointIdx,0], np.array(points)[pointIdx,1]))
            else:
                # Redraw current image and any previously entered points for context
                plt.figure(figsize=[8,8])
                plt.pcolormesh(
                    OCTAL.x, OCTAL.y,
                    np.where(_temp_mask, sample_avg_darkbackmask, np.nan),
                    shading='auto'
                )
                plt.axis('equal')
                plt.minorticks_on()
                plt.grid(True, which='both')
                plt.title(f'Enter point {pointIdx+1}/{num_points} (x, y)')
                if entered_points:
                    xs, ys = zip(*entered_points)
                    plt.plot(xs, ys, 'ro-', ms=4, lw=1, label='Entered points')
                plt.show()
            
                # Prompt for coordinates
                x_pt = float(input(f'Enter x-coord of point {pointIdx+1}: '))
                y_pt = float(input(f'Enter y-coord of point {pointIdx+1}: '))
                entered_points.append((x_pt, y_pt))
        
        # Close polygon and build mask
        verts = np.asarray(entered_points, dtype=float)
        verts_closed = np.vstack([verts, verts[0]])  # close loop
        codes = np.full(len(verts_closed), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        
        polygon = Path(verts_closed, codes)
        grid_pts = np.column_stack([OCTAL.x.ravel(), OCTAL.y.ravel()])
        inside = polygon.contains_points(grid_pts).reshape(OCTAL.x.shape)
        
        # Keep inside region, mask outside
        self.poly_mask = inside.astype(bool)
        
        # Apply to the sample_avg_darkbackmask visualization
        sample_avg_darkbackpolymask = np.where(self.poly_mask, sample_avg_darkbackmask, np.nan)
        
        # Show final masked result
        plt.pcolormesh(OCTAL.x, OCTAL.y, sample_avg_darkbackpolymask, shading='auto')
        plt.axis('image')
        plt.title("Masked result (outside polygon set to NaN)")
        plt.show()
        

        #### --- Figure out q-grid for slice maskking ---  ####
        
        # Determine q values. Maybe this should be in its own function?
        # --- 2. Geometric Transformations ---
        tx_rad, ty_rad = np.deg2rad(tx), np.deg2rad(ty)
        z_total = z0 + z_off

        # Geometric parameters from J Chem Phys 113, 9140 (2000)
        A = -np.sin(ty_rad) * np.cos(tx_rad)
        B = -np.sin(tx_rad)
        C = -np.cos(ty_rad) * np.cos(tx_rad)
        a = x0 + z_total * np.tan(ty_rad)
        b = y0 - z_total * np.tan(tx_rad)
        c = z_total

        # Transforming (x,y) to r, theta, phi
        R = np.sqrt((OCTAL.x - a) ** 2 + (OCTAL.y - b) ** 2 + c ** 2)
        Matrix_theta = np.arccos((A * (OCTAL.x - a) + B * (OCTAL.y - b) - C * c) / R)
        # Binning in reciprocal space (q)
        lam = keV2Angstroms(keV)
        Radial_map = 4 * np.pi / lam * np.sin(Matrix_theta / 2)
        q_min = np.nanmin(Radial_map)
        q_max = np.nanmax(Radial_map)
        q_bins = np.linspace(q_min, q_max, 200)

        # Preallocation
        mean_intensity = np.zeros_like(q_bins)
        std_intensity = np.zeros_like(q_bins)
        
        # plt.errorbar(q_bins, mean_intensity, yerr=std_intensity)
        # plt.xlabel('q')
        # plt.ylabel('Slice mean & std')
        # plt.minorticks_on()
        # plt.grid(True, which='both')
        # plt.show()
        # # User input
        # user_lb_q = float(input('Enter lower bound for usable q-bins (cannot = 0): '))
        # user_ub_q = float(input('Enter upper bound for usable q-bins: '))
            

        # # Loop through the q bins
        for qidx,_q in enumerate(tqdm(q_bins[:-1])):
            # First create a ring mask for this q
            Ring_mask = (Radial_map >= q_bins[qidx]) * (Radial_map < q_bins[qidx+1])
            Q_ring = np.copy(sample_avg_darkbackpolymask)
            Q_ring[~Ring_mask] = np.nan

            # Check to see if all of the pixels are already masked before proceeding.
            if ~np.all(Q_ring==np.nan):

                mean_intensity[qidx] = np.nanmean(Q_ring)
                std_intensity[qidx] = np.nanstd(Q_ring)
                # Figure out mask for this ring based on n*std
                lower_bound = mean_intensity[qidx] - n_std * std_intensity[qidx]
                upper_bound = mean_intensity[qidx] + n_std * std_intensity[qidx]
    
                # Figure out the pixels that are outside the bound in this ring, then mask them
                this_ring_mask = np.ones_like(Q_ring, dtype=bool)
                this_ring_mask[~np.isnan(Q_ring)] = (Q_ring[~np.isnan(Q_ring)] >= lower_bound) & (Q_ring[~np.isnan(Q_ring)] <= upper_bound)
    
                # Masking
                # if (_q < user_lb_q) or (_q > user_ub_q):
                #     # Skip this ring, set it to all nan.
                #     self.sample_mask[Ring_mask] = 0
                
                
                if np.sum(this_ring_mask[Ring_mask]) / np.sum(Ring_mask) > 0.95 and qidx != 0:
                    # If more than 95% of the pixels are kept, automatically apply the mask
                    if self.verbose:
                        print(f'Q bin {_q:.3f}: automatically masked percent {100*(1-np.sum(this_ring_mask[Ring_mask])/np.sum(Ring_mask)):.2f}%')
                    self.sample_mask *= this_ring_mask
                elif qidx != 0:
                    # Otherwise, plot the histogram and ask for manual input
                    print('Manual check needed for q bin ', qidx)
    
                    nanmask = np.isnan(Q_ring)
                    # Plot the histogram
                    plt.figure(figsize=(12,6))
                    plt.subplot(1,2,1)
                    plt.hist(Q_ring[Ring_mask *(~nanmask)].flatten(), bins=200, range=(np.nanpercentile(Q_ring[Ring_mask].flatten(),0.5),np.nanpercentile(Q_ring[Ring_mask].flatten(),99.5)))
                    plt.axvline(mean_intensity[qidx],color='r',linestyle='--')
                    plt.axvline(lower_bound,color='k',linestyle='--')
                    plt.axvline(upper_bound,color='k',linestyle='--')
                    plt.title(f'Run {self.sample_run_number} Sample Intensity Histogram for range {q_bins[qidx]:.3f} to {q_bins[qidx+1]:.3f}')
                    plt.xlabel('Intensity')
                    plt.ylabel('Counts')
                    plt.grid(True)
    
                    # Plot the actual ring, cropped to the area that it covers.
                    xmin = np.nanmin(OCTAL.x[Ring_mask])
                    xmax = np.nanmax(OCTAL.x[Ring_mask])
                    ymin = np.nanmin(OCTAL.y[Ring_mask])
                    ymax = np.nanmax(OCTAL.y[Ring_mask])
                    plt.subplot(1,2,2)
                    pcm = plt.pcolormesh(OCTAL.x, OCTAL.y, Q_ring)
                    plt.axis('image')
                    plt.xlim([xmin, xmax])
                    plt.ylim([ymin, ymax])
                    plt.title(f'Ring {q_bins[qidx]:.3f} to {q_bins[qidx+1]:.3f}')
                    plt.colorbar(pcm, fraction=0.046, pad=0.04)
                    plt.show()
    
                    # User input
                    user_lb = float(input('Enter lower bound for sample mask: '))
                    user_ub = float(input('Enter upper bound for sample mask: '))
                    ring_mask_manual = np.ones_like(Q_ring, dtype=bool)
                    ring_mask_manual[~np.isnan(Q_ring)] = (Q_ring[~np.isnan(Q_ring)] >= user_lb) * (Q_ring[~np.isnan(Q_ring)] <= user_ub)
                    self.sample_mask *= ring_mask_manual
                else:
                    # Bad. Set everything to zero yeaaa
                    self.sample_mask[Ring_mask] = 0

                        
        # Convert to boolean
        self.sample_mask = self.sample_mask.astype(bool)
        if plotting:
            fig2, ax2 = plt.subplots(ncols=3, figsize=(18, 8))

            sample_avg_backdarksamplemasked = np.copy(sample_avg_darkbackpolymask)
            sample_avg_backdarksamplemasked[~(self.sample_mask)] = np.nan

            # Plot background mask
            pcm = ax2[0].imshow(self.sample_mask,vmin=0, vmax=1)
            ax2[0].set_title('Mask from Sample')
            ax2[0].set_xticks([])
            ax2[0].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[0], fraction=0.046, pad=0.04)
            # Plot arcsinh background average with dark mask applied
            pcm = ax2[1].imshow(sample_avg_darkbackpolymask,cmap='jet',vmin=0)
            ax2[1].set_title('arcsinh(Dark Masked Bkg Average)')
            ax2[1].set_xticks([])
            ax2[1].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[1], fraction=0.046, pad=0.04)
            # Plot masked background average
            pcm = ax2[2].imshow(sample_avg_backdarksamplemasked,cmap='jet',vmin=0)
            ax2[2].set_title('Masked Bkg Average')
            ax2[2].set_xticks([])
            ax2[2].set_yticks([])
            fig2.colorbar(pcm, ax=ax2[2], fraction=0.046, pad=0.04)
            plt.show()
        # Reporting masked percentage
        print(f'Masked percentage from sample: {100*(1-np.sum(self.sample_mask)/self.sample_mask.size):.2f}%')
        pass

    def combine_masks(self, plotting=True):
        """
        Combine the dark, background, and sample masks into a single mask.

        Parameters
        ----------
        plotting : bool, optional
            Whether to plot the combined mask and its components (default is True).

        Returns
        -------
        None
        """
        # self.cmask = (self.dark_mask * self.background_mask * self.sample_mask * J4M.line_mask * J4M.t_mask).astype(bool)
        self.cmask = (self.dark_mask * self.background_mask * self.sample_mask * self.poly_mask).astype(bool)
        if plotting:
            # Grabbing the data from the sample_data attribute, calculating average
            # sample_num_shots = np.where(self.sample_data['lightStatus/xray'].astype(bool)==True)[0].shape[0]
            # sample_avg = np.array(self.sample_data['Sums/jungfrau4M_calib_xrayOn_thresADU1'] / sample_num_shots)

            sample_avg_combinedmask = np.copy(self.sample_data)
            sample_avg_combinedmask[~(self.cmask)] = np.nan

            fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
            pcm = ax[0,0].imshow(self.dark_mask,vmin=0, vmax=1)
            ax[0,0].set_title('Dark Mask')
            ax[0,0].set_xticks([])
            ax[0,0].set_yticks([])
            fig.colorbar(pcm, ax=ax[0,0], fraction=0.046, pad=0.04)
            pcm = ax[0,1].imshow(self.background_mask,vmin=0, vmax=1)
            ax[0,1].set_title('Background Mask')
            ax[0,1].set_xticks([])
            ax[0,1].set_yticks([])
            fig.colorbar(pcm, ax=ax[0,1], fraction=0.046, pad=0.04)
            pcm = ax[1,0].imshow(self.poly_mask,vmin=0, vmax=1)
            ax[1,0].set_title('Polygon Mask')
            ax[1,0].set_xticks([])
            ax[1,0].set_yticks([])
            fig.colorbar(pcm, ax=ax[1,0], fraction=0.046, pad=0.04)
            pcm = ax[1,1].imshow(self.sample_mask,vmin=0, vmax=1)
            ax[1,1].set_title('Sample Mask')
            ax[1,1].set_xticks([])
            ax[1,1].set_yticks([])
            fig.colorbar(pcm, ax=ax[1,1], fraction=0.046, pad=0.04)
            plt.show()

            fig, ax = plt.subplots(figsize=(10,10))
            pcm = ax.imshow(self.cmask, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Combined Mask')
            plt.show()

            fig, ax = plt.subplots(figsize=(10,10))
            pcm = ax.imshow(sample_avg_combinedmask, vmin=0, cmap='jet')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Sample Average with Combined Mask Applied')
            plt.show()
        print(f'Total masked percentage: {100*(1-np.sum(self.cmask)/self.cmask.size):.2f}%')
        pass

    def save_mask(self, valid_from_run=None, mask_directory=None):
        """
        Save the combined mask to a file.

        Parameters
        ----------
        valid_from_run : int, optional
            The run number from which the mask is valid (default is background_run_number).
        mask_directory : str, optional
            Directory to save the mask file (default is a predefined path based on the experiment).

        Returns
        -------
        None
        """
        # Determining when the mask starts being valid
        if valid_from_run is None:
            valid_from_run = self.background_run_number
        # Making the file name and path
        if mask_directory is None:
            mask_directory = f'Masks/'
        os.makedirs(mask_directory, exist_ok=True)
        mask_filename = f'{valid_from_run}-end.h5'
        with h5py.File(mask_directory+mask_filename, "w") as f:
            f.create_dataset("mask", data=self.cmask)

        print(f'Saved combined mask to {mask_directory+mask_filename}')
        pass
    
