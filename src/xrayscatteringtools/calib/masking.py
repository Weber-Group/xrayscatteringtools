from __future__ import annotations

from typing import Optional

from ..utils import J4M, keV2Angstroms, compute_q_map
from ..io import combineRuns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Patch
from ..plotting import plot_j4m
from .geometry_calibration import thompson_correction
from tqdm.auto import tqdm
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants used across masking methods
# ---------------------------------------------------------------------------
_KEYS_TO_COMBINE = [
    'lightStatus/xray',
    'lightStatus/laser',
    'jungfrau4M/azav_azav',
]
_KEYS_TO_SUM = [
    'Sums/jungfrau4M_calib_xrayOn_thresADU1',
    'Sums/jungfrau4M_calib_dropped',
]
_KEYS_TO_CHECK: list[str] = []

# Default percentile window for recommended dark bounds
_DARK_PERCENTILE_LO = 0.5
_DARK_PERCENTILE_HI = 99.5
# Default percentile for recommended background upper bound
_BKG_PERCENTILE_HI = 99
# Number of q-bins used in the sample ring masking
_N_Q_BINS = 100
# Fraction of pixels that must survive the n-sigma cut for auto-accept
_AUTO_ACCEPT_THRESHOLD = 0.95


class MaskMaker:
    """Create and manage masks for Jungfrau 4M X-ray scattering data.

    The workflow is:
        1. Instantiate with experiment info and run numbers.
        2. Call :meth:`process_dark` to mask hot / dead pixels.
        3. Call :meth:`process_background` to mask background outliers.
        4. (Optional) Call :meth:`apply_polygon_mask` to manually exclude regions.
        5. (Optional) Call :meth:`diagnose_q_bins` to preview per-ring statistics.
        6. Call :meth:`process_sample` to mask per-ring intensity outliers.
        7. Call :meth:`combine_masks` to merge all masks.
        8. Call :meth:`save_mask` to persist the result.

    Parameters
    ----------
    experiment : str
        The LCLS experiment identifier (e.g. ``'cxilv4418'``).
    data_path : str
        Path to the directory containing the HDF5 run files.
    dark_run_number : int
        Run number for the dark (no-beam) data.
    background_run_number : int
        Run number for the background (beam-on, no sample) data.
    sample_run_number : int
        Run number for the sample data.
    verbose : bool, optional
        If ``True``, print extra diagnostics during processing.

    Attributes
    ----------
    dark_mask : np.ndarray
        Boolean mask derived from dark data.
    background_mask : np.ndarray
        Boolean mask derived from background data.
    poly_mask : np.ndarray
        Boolean mask from user-drawn polygon(s).
    sample_mask : np.ndarray
        Boolean mask derived from per-ring sample statistics.
    cmask : np.ndarray
        Combined (final) boolean mask.
    """

    def __init__(
        self,
        experiment: str,
        data_path: str,
        dark_run_number: int,
        background_run_number: int,
        sample_run_number: int,
        verbose: bool = False,
    ) -> None:
        # Validate inputs
        if not isinstance(experiment, str) or not experiment:
            raise ValueError("'experiment' must be a non-empty string.")
        if not isinstance(data_path, str) or not data_path:
            raise ValueError("'data_path' must be a non-empty string.")
        for name, val in [
            ("dark_run_number", dark_run_number),
            ("background_run_number", background_run_number),
            ("sample_run_number", sample_run_number),
        ]:
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValueError(f"'{name}' must be a non-negative integer, got {val!r}.")

        # Setting class attributes
        self.verbose: bool = verbose
        self.experiment: str = experiment
        self.data_path: str = data_path
        self.dark_run_number: int = int(dark_run_number)
        self.background_run_number: int = int(background_run_number)
        self.sample_run_number: int = int(sample_run_number)

        # Loading data
        load_kw = dict(verbose=verbose)
        self.dark_data: dict = combineRuns(
            dark_run_number, data_path, _KEYS_TO_COMBINE, _KEYS_TO_SUM, _KEYS_TO_CHECK, **load_kw
        )
        self.background_data: dict = combineRuns(
            background_run_number, data_path, _KEYS_TO_COMBINE, _KEYS_TO_SUM, _KEYS_TO_CHECK, **load_kw
        )
        self.sample_data: dict = combineRuns(
            sample_run_number, data_path, _KEYS_TO_COMBINE, _KEYS_TO_SUM, _KEYS_TO_CHECK, **load_kw
        )

        # Initialize masks (all pixels valid)
        self.dark_mask: np.ndarray = np.ones_like(J4M.x, dtype=bool)
        self.background_mask: np.ndarray = np.ones_like(J4M.x, dtype=bool)
        self.poly_mask: np.ndarray = np.ones_like(J4M.x, dtype=bool)
        self.sample_mask: np.ndarray = np.ones_like(J4M.x, dtype=bool)
        self.cmask: np.ndarray = np.ones_like(J4M.x, dtype=bool)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MaskMaker(experiment={self.experiment!r}, "
            f"dark={self.dark_run_number}, bkg={self.background_run_number}, "
            f"sample={self.sample_run_number})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_fraction(mask: np.ndarray) -> float:
        """Return the fraction of pixels that are masked (``False``)."""
        return 1.0 - np.sum(mask) / mask.size

    @staticmethod
    def _plot_j4m_panel(
        ax: plt.Axes,
        data: np.ndarray,
        title: str,
        fig: plt.Figure,
        **kwargs,
    ) -> None:
        """Plot a single J4M panel with colorbar and clean axis labels."""
        pcm = plot_j4m(data, ax=ax, **kwargs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    def _compute_sample_average(self, apply_masks: bool = True) -> np.ndarray:
        """Compute the average sample image, optionally applying current masks.

        Parameters
        ----------
        apply_masks : bool
            If ``True``, pixels failing *dark*, *background*, and *polygon*
            masks are set to ``NaN``.

        Returns
        -------
        np.ndarray
            Average sample image.
        """
        n_shots = np.sum(self.sample_data['lightStatus/xray'].astype(bool))
        avg = np.array(self.sample_data['Sums/jungfrau4M_calib_xrayOn_thresADU1'] / n_shots)
        if apply_masks:
            combined = self.dark_mask & self.background_mask & self.poly_mask
            avg = np.where(combined, avg, np.nan)
        return avg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_dark(
        self,
        plotting: bool = True,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ) -> None:
        """Create a mask from the dark (no-beam) data.

        Pixels whose ``arcsinh(intensity)`` falls outside ``[lb, ub]`` are
        masked.  If bounds are not supplied, the user is prompted
        interactively (with recommended defaults shown).

        Parameters
        ----------
        plotting : bool, optional
            Show diagnostic histograms and mask images.
        lb : float, optional
            Lower ``arcsinh`` intensity cutoff.
        ub : float, optional
            Upper ``arcsinh`` intensity cutoff.
        """
        # Compute average dark image (xray-off frames only)
        dark_num_shots = np.sum(~self.dark_data['lightStatus/xray'].astype(bool))
        dark_avg = np.array(self.dark_data['Sums/jungfrau4M_calib_dropped'] / dark_num_shots)
        dark_arcsinh = np.arcsinh(dark_avg)

        if plotting:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            for ax, yscale in zip(axes, ['log', 'linear']):
                ax.hist(dark_arcsinh.flatten(), bins=500)
                ax.set_title(f'Run {self.dark_run_number} Dark Image Histogram')
                ax.set_yscale(yscale)
                ax.set_xlabel('arcsinh(Intensity)')
                ax.set_ylabel('Counts')
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        # Determine recommended bounds
        recommended_bounds = np.percentile(dark_arcsinh.flatten(), [_DARK_PERCENTILE_LO, _DARK_PERCENTILE_HI])
        if lb is None or ub is None:
            print('Determine intensity cutoffs for dark mask (leave blank to use recommended bounds):')
            lb = float(input(f'Enter lower bound for dark mask (recommended: {recommended_bounds[0]:.4f}): ') or recommended_bounds[0])
            ub = float(input(f'Enter upper bound for dark mask (recommended: {recommended_bounds[1]:.4f}): ') or recommended_bounds[1])
        else:
            lb, ub = float(lb), float(ub)

        if lb >= ub:
            raise ValueError(f"Lower bound ({lb}) must be less than upper bound ({ub}).")

        # Build the mask
        self.dark_mask = (dark_arcsinh >= lb) & (dark_arcsinh <= ub)

        if plotting:
            dark_avg_masked = np.where(self.dark_mask, dark_avg, np.nan)

            fig, axes = plt.subplots(1, 3, figsize=(18, 8))
            self._plot_j4m_panel(axes[0], self.dark_mask, 'Mask from Dark', fig, vmin=0, vmax=1)
            self._plot_j4m_panel(axes[1], dark_arcsinh, 'arcsinh(Dark Average)', fig, cmap='jet')
            self._plot_j4m_panel(axes[2], dark_avg_masked, 'Masked Dark Average', fig, cmap='jet')
            plt.tight_layout()
            plt.show()

        print(f'Masked percentage from dark: {100 * self._masked_fraction(self.dark_mask):.2f}%')

    def process_background(
        self,
        plotting: bool = True,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ) -> None:
        """Create a mask from the background (beam-on, no sample) data.

        The dark mask is applied first, then pixels outside ``[lb, ub]`` are
        masked.  If bounds are not supplied the user is prompted
        interactively.

        Parameters
        ----------
        plotting : bool, optional
            Show diagnostic histograms and mask images.
        lb : float, optional
            Lower intensity cutoff.
        ub : float, optional
            Upper intensity cutoff.
        """
        # Compute average background image (xray-on frames)
        bkg_num_shots = np.sum(self.background_data['lightStatus/xray'].astype(bool))
        bkg_avg = np.array(self.background_data['Sums/jungfrau4M_calib_xrayOn_thresADU1'] / bkg_num_shots)
        bkg_avg_darkmask = np.where(self.dark_mask, bkg_avg, np.nan)

        if plotting:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            data_range = (0, np.nanmax(bkg_avg_darkmask))
            for ax, yscale in zip(axes, ['log', 'linear']):
                ax.hist(bkg_avg_darkmask.flatten(), bins=500, range=data_range)
                ax.set_title(f'Run {self.background_run_number} Background Image Histogram')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('Counts')
                ax.set_yscale(yscale)
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        # Determine recommended bounds
        recommended_bounds = [0, np.nanpercentile(bkg_avg_darkmask.flatten(), _BKG_PERCENTILE_HI)]
        if lb is None or ub is None:
            print('Determine intensity cutoffs for background mask (leave blank to use recommended bounds):')
            lb = float(input(f'Enter lower bound for background mask (recommended: {recommended_bounds[0]:.4f}): ') or recommended_bounds[0])
            ub = float(input(f'Enter upper bound for background mask (recommended: {recommended_bounds[1]:.4f}): ') or recommended_bounds[1])
        else:
            lb, ub = float(lb), float(ub)

        if lb >= ub:
            raise ValueError(f"Lower bound ({lb}) must be less than upper bound ({ub}).")

        # Build the mask (NaN-safe comparison)
        bkg_filled = np.nan_to_num(bkg_avg_darkmask)
        self.background_mask = (bkg_filled >= lb) & (bkg_filled <= ub)

        if plotting:
            bkg_avg_backmask = np.where(self.background_mask, bkg_avg_darkmask, np.nan)

            fig, axes = plt.subplots(1, 3, figsize=(18, 8))
            self._plot_j4m_panel(axes[0], self.background_mask, 'Mask from Bkg', fig, vmin=0, vmax=1)
            self._plot_j4m_panel(axes[1], np.arcsinh(bkg_avg_darkmask), 'arcsinh(Dark Masked Bkg Average)', fig, cmap='jet', vmin=0)
            self._plot_j4m_panel(axes[2], bkg_avg_backmask, 'Masked Bkg Average', fig, cmap='jet', vmin=0)
            plt.tight_layout()
            plt.show()

        print(f'Masked percentage from background: {100 * self._masked_fraction(self.background_mask):.2f}%')

    def apply_polygon_mask(
        self,
        num_points: int,
        points: Optional[list[tuple[float, float]]] = None,
        plotting: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Draw a polygon mask interactively or from supplied vertices.

        Pixels *outside* the polygon are masked.  Uses ``plot_j4m`` display
        coordinates — note that these require a ``(y, -x)`` flip relative to
        the raw J4M pixel grid, which is handled automatically.

        Parameters
        ----------
        num_points : int
            Number of polygon vertices (must be >= 3).
        points : list of (x, y) tuples, optional
            Pre-defined vertices.  If fewer than *num_points* are given the
            user is prompted for the remainder.
        plotting : bool, optional
            Show the masked result.
        *args, **kwargs
            Forwarded to :func:`plot_j4m`.

        Notes
        -----
        If *points* are not (fully) provided, the user is prompted to enter
        coordinates one by one, with an updated plot shown after each entry
        for spatial context.  The polygon is automatically closed.
        """
        if num_points < 3:
            raise ValueError("At least 3 points are required to form a polygon.")
        if points is None:
            points = []

        # Reset the polygon mask
        self.poly_mask = np.ones_like(J4M.x, dtype=bool)

        # Compute masked sample average for the context plot
        sample_avg_masked = self._compute_sample_average(apply_masks=True)

        entered_points: list[tuple[float, float]] = []
        print('Beginning polygon masking')

        for point_idx in range(num_points):
            if point_idx < len(points):
                pt = points[point_idx]
                entered_points.append((float(pt[0]), float(pt[1])))
            else:
                # Show context plot with previously entered points
                plt.figure(figsize=(8, 8))
                plot_j4m(sample_avg_masked, *args, **kwargs)
                plt.axis('equal')
                plt.minorticks_on()
                plt.grid(True, which='both')
                plt.title(f'Enter point {point_idx + 1}/{num_points} (x, y)')
                plt.xlabel('x (mm)')
                plt.ylabel('y (mm)')
                if entered_points:
                    xs, ys = zip(*entered_points)
                    plt.plot(xs, ys, 'ro-', ms=4, lw=1, label='Entered points')
                plt.show()

                x_pt = float(input(f'Enter x-coord of point {point_idx + 1}: '))
                y_pt = float(input(f'Enter y-coord of point {point_idx + 1}: '))
                entered_points.append((x_pt, y_pt))

        # Flip coordinates to match the (y, -x) convention used by plot_j4m
        entered_points_flipped = [(-y, x) for x, y in entered_points]

        # Build closed polygon path
        verts = np.asarray(entered_points_flipped, dtype=float)
        verts_closed = np.vstack([verts, verts[0]])
        codes = np.full(len(verts_closed), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        polygon = Path(verts_closed, codes)
        grid_pts = np.column_stack([J4M.x.ravel(), J4M.y.ravel()])
        inside = polygon.contains_points(grid_pts).reshape(J4M.x.shape)

        # Keep inside, mask outside
        self.poly_mask = inside.astype(bool)

        if plotting:
            sample_avg_poly = np.where(self.poly_mask, sample_avg_masked, np.nan)
            plt.figure(figsize=(8, 8))
            plot_j4m(sample_avg_poly, *args, **kwargs)
            plt.axis('image')
            plt.title('Masked result (outside polygon set to NaN)')
            plt.show()

    def process_sample(
        self,
        n_std: float = 2,
        x0: float = 0,
        y0: float = 0,
        z0: float = 90_000,
        tx: float = 0,
        ty: float = 0,
        keV: float = 10,
        z_off: float = 0,
        phi0: float = 0,
        n_q_bins: int = _N_Q_BINS,
        auto_accept_threshold: float = _AUTO_ACCEPT_THRESHOLD,
        plotting: bool = True,
    ) -> None:
        """Create a mask from per-ring intensity statistics on the sample data.

        For each momentum-transfer (*q*) ring the mean and standard deviation
        are computed.  Pixels further than *n_std* standard deviations from
        the mean are flagged.  If fewer than ``(1 - auto_accept_threshold)``
        pixels are removed the ring is accepted automatically; otherwise the
        user is prompted to set bounds manually.

        A Thompson (polarization) correction is applied before the per-ring
        analysis so that the azimuthal intensity variation from polarised
        X-rays does not bias the statistics.

        Parameters
        ----------
        n_std : float, optional
            Number of standard deviations for the automatic intensity cutoff.
        x0, y0 : float, optional
            Beam-centre coordinates in microns.
        z0 : float, optional
            Sample-to-detector distance in microns.
        tx, ty : float, optional
            Detector tilt angles in degrees.
        keV : float, optional
            Photon energy in keV.
        z_off : float, optional
            Additional z offset in microns.
        phi0 : float, optional
            Azimuthal rotation angle (radians) for the Thompson polarization
            correction.  Default is 0 (horizontal polarization).
        n_q_bins : int, optional
            Number of q bins for ring-by-ring masking.
        auto_accept_threshold : float, optional
            Fraction of surviving pixels above which a ring is auto-accepted.
        plotting : bool, optional
            Show diagnostic plots.
        """
        self.sample_mask = np.ones_like(J4M.x, dtype=bool)

        # Average sample image with current masks applied
        sample_avg_masked = self._compute_sample_average(apply_masks=True)

        # Apply Thompson (polarization) correction
        centered_x = J4M.x - x0
        centered_y = J4M.y - y0
        thompson_corr = thompson_correction(centered_x, centered_y, z0 + z_off, phi0)
        sample_avg_corrected = sample_avg_masked / thompson_corr

        # Compute the q-map
        q_map = compute_q_map(J4M.x, J4M.y, x0, y0, z0, tx, ty, keV, z_off)
        q_bins = np.linspace(np.nanmin(q_map), np.nanmax(q_map), n_q_bins)

        for q_idx, q_lo in tqdm(enumerate(q_bins[:-1]), total=len(q_bins) - 1):
            q_hi = q_bins[q_idx + 1]
            ring_sel = (q_map >= q_lo) & (q_map < q_hi)
            q_ring = np.where(ring_sel, sample_avg_corrected, np.nan)

            mean_i = np.nanmean(q_ring)
            std_i = np.nanstd(q_ring)
            lower_bound = mean_i - n_std * std_i
            upper_bound = mean_i + n_std * std_i

            ring_mask = np.ones_like(q_ring, dtype=bool)
            valid = ~np.isnan(q_ring)
            ring_mask[valid] = (q_ring[valid] >= lower_bound) & (q_ring[valid] <= upper_bound)

            kept_frac = np.sum(ring_mask[ring_sel]) / np.sum(ring_sel)
            if kept_frac > auto_accept_threshold and q_idx != 0:
                if self.verbose:
                    logger.info(
                        'Q bin %.3f: auto-masked %.2f%%',
                        q_lo,
                        100 * (1 - kept_frac),
                    )
                self.sample_mask &= ring_mask
            else:
                # Manual review required — show histogram and detector ring
                ring_data = q_ring[ring_sel].flatten()
                print(f'Manual check needed for q bin {q_idx}')

                fig, (ax_hist, ax_ring) = plt.subplots(1, 2, figsize=(16, 6))

                # Left: intensity histogram
                ax_hist.hist(
                    ring_data,
                    bins=200,
                    range=(np.nanpercentile(ring_data, 0.5),
                           np.nanpercentile(ring_data, 99.5)),
                )
                ax_hist.set_title(
                    f'Run {self.sample_run_number} Intensity Histogram\n'
                    f'q = [{q_lo:.3f}, {q_hi:.3f})'
                )
                ax_hist.set_xlabel('Intensity')
                ax_hist.set_ylabel('Counts')
                ax_hist.grid(True)

                # Right: pcolormesh of the ring pixels on the detector
                ring_display = np.where(ring_sel, sample_avg_corrected, np.nan)
                pcm = plot_j4m(ring_display, ax=ax_ring, cmap='jet')
                fig.colorbar(pcm, ax=ax_ring, fraction=0.046, pad=0.04)
                ax_ring.set_title(f'Q-ring pixels: q = [{q_lo:.3f}, {q_hi:.3f})')

                # Zoom to the bounding box of the ring
                # plot_j4m uses display coords (J4M.y, -J4M.x)
                display_x_vals = []
                display_y_vals = []
                for tile in range(8):
                    tile_ring = ring_sel[tile]
                    if np.any(tile_ring):
                        display_x_vals.append(J4M.y[tile][tile_ring])
                        display_y_vals.append(-J4M.x[tile][tile_ring])
                if display_x_vals:
                    all_dx = np.concatenate(display_x_vals)
                    all_dy = np.concatenate(display_y_vals)
                    margin = 5000  # microns
                    ax_ring.set_xlim(all_dx.min() - margin, all_dx.max() + margin)
                    ax_ring.set_ylim(all_dy.min() - margin, all_dy.max() + margin)
                ax_ring.set_aspect('equal')

                plt.tight_layout()
                plt.show()

                user_lb = float(input('Enter lower bound for sample mask: '))
                user_ub = float(input('Enter upper bound for sample mask: '))
                manual_mask = np.ones_like(q_ring, dtype=bool)
                manual_mask[valid] = (q_ring[valid] >= user_lb) & (q_ring[valid] <= user_ub)
                self.sample_mask &= manual_mask

        self.sample_mask = self.sample_mask.astype(bool)

        if plotting:
            sample_avg_all_masked = np.where(self.sample_mask, sample_avg_masked, np.nan)

            fig, axes = plt.subplots(1, 3, figsize=(18, 8))
            self._plot_j4m_panel(axes[0], self.sample_mask, 'Mask from Sample', fig, vmin=0, vmax=1)
            self._plot_j4m_panel(axes[1], sample_avg_masked, 'Drk., Bkg., Poly. Masked Sample Avg', fig, cmap='jet', vmin=0)
            self._plot_j4m_panel(axes[2], sample_avg_all_masked, 'Drk., Bkg., Poly., Samp. Masked Sample Avg', fig, cmap='jet', vmin=0)
            plt.tight_layout()
            plt.show()

        print(f'Masked percentage from sample: {100 * self._masked_fraction(self.sample_mask):.2f}%')

    def diagnose_q_bins(
        self,
        n_std: float = 2,
        x0: float = 0,
        y0: float = 0,
        z0: float = 90_000,
        tx: float = 0,
        ty: float = 0,
        keV: float = 10,
        z_off: float = 0,
        phi0: float = 0,
        n_q_bins: int = _N_Q_BINS,
        auto_accept_threshold: float = _AUTO_ACCEPT_THRESHOLD,
    ) -> None:
        """Preview per-q-bin statistics before running :meth:`process_sample`.

        Produces an error-bar chart where each bar represents the mean
        Thompson-corrected intensity in a q bin, with the standard deviation
        as the error bar.  Bins that would require manual review (kept
        fraction <= *auto_accept_threshold* or the first bin) are shown in
        red; auto-accepted bins are shown in steel-blue.

        This is useful for choosing *n_std* and *auto_accept_threshold*
        values before committing to the interactive ``process_sample`` loop.

        Parameters
        ----------
        n_std : float, optional
            Number of standard deviations for the automatic intensity cutoff.
        x0, y0 : float, optional
            Beam-centre coordinates in microns.
        z0 : float, optional
            Sample-to-detector distance in microns.
        tx, ty : float, optional
            Detector tilt angles in degrees.
        keV : float, optional
            Photon energy in keV.
        z_off : float, optional
            Additional z offset in microns.
        phi0 : float, optional
            Azimuthal rotation angle (radians) for the Thompson polarization
            correction.
        n_q_bins : int, optional
            Number of q bins.
        auto_accept_threshold : float, optional
            Fraction of surviving pixels above which a ring is auto-accepted.
        """
        sample_avg_masked = self._compute_sample_average(apply_masks=True)

        # Apply Thompson (polarization) correction
        centered_x = J4M.x - x0
        centered_y = J4M.y - y0
        thompson_corr = thompson_correction(centered_x, centered_y, z0 + z_off, phi0)
        sample_avg_corrected = sample_avg_masked / thompson_corr

        q_map = compute_q_map(J4M.x, J4M.y, x0, y0, z0, tx, ty, keV, z_off)
        q_bins = np.linspace(np.nanmin(q_map), np.nanmax(q_map), n_q_bins)
        q_centers = (q_bins[:-1] + q_bins[1:]) / 2

        means = np.empty(len(q_centers))
        stds = np.empty(len(q_centers))
        needs_review = np.zeros(len(q_centers), dtype=bool)

        for q_idx in range(len(q_centers)):
            ring_sel = (q_map >= q_bins[q_idx]) & (q_map < q_bins[q_idx + 1])
            q_ring = np.where(ring_sel, sample_avg_corrected, np.nan)

            mean_i = np.nanmean(q_ring)
            std_i = np.nanstd(q_ring)
            means[q_idx] = mean_i
            stds[q_idx] = std_i

            # Determine whether this bin would need manual review
            lower_bound = mean_i - n_std * std_i
            upper_bound = mean_i + n_std * std_i
            ring_mask = np.ones_like(q_ring, dtype=bool)
            valid = ~np.isnan(q_ring)
            ring_mask[valid] = (q_ring[valid] >= lower_bound) & (q_ring[valid] <= upper_bound)
            kept_frac = np.sum(ring_mask[ring_sel]) / np.sum(ring_sel)
            if kept_frac <= auto_accept_threshold or q_idx == 0:
                needs_review[q_idx] = True

        # --- Plot ---
        colors = np.where(needs_review, 'tab:red', 'steelblue')
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(
            q_centers,
            means,
            width=np.diff(q_bins),
            yerr=stds,
            color=colors,
            edgecolor='k',
            linewidth=0.3,
            capsize=2,
            alpha=0.8,
            ecolor='gray',
        )
        ax.set_xlabel(r'$q\;(\AA^{-1})$')
        ax.set_ylabel('Mean Intensity (Thompson-corrected)')
        ax.set_title(
            f'Q-bin diagnostics  (n_std={n_std}, '
            f'auto_accept_threshold={auto_accept_threshold})'
        )
        legend_elements = [
            Patch(facecolor='steelblue', edgecolor='k', label='Auto-accepted'),
            Patch(facecolor='tab:red', edgecolor='k', label='Manual review needed'),
        ]
        ax.legend(handles=legend_elements)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        n_review = int(np.sum(needs_review))
        print(
            f'{n_review}/{len(q_centers)} q-bins would require manual review '
            f'with current settings.'
        )

    def combine_masks(self, plotting: bool = True) -> None:
        """Combine all individual masks into :attr:`cmask`.

        The combined mask is the element-wise AND of the dark, background,
        polygon, sample, line, and T masks.

        Parameters
        ----------
        plotting : bool, optional
            Show diagnostic plots of each component and the final mask.
        """
        self.cmask = (
            self.dark_mask
            & self.background_mask
            & self.poly_mask
            & self.sample_mask
            & J4M.line_mask.astype(bool)
            & J4M.t_mask.astype(bool)
        )

        if plotting:
            sample_avg = self._compute_sample_average(apply_masks=False)
            sample_avg_combined = np.where(self.cmask, sample_avg, np.nan)

            # Component masks overview
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
            mask_panels = [
                (ax[0, 0], self.dark_mask, 'Dark Mask'),
                (ax[0, 1], self.background_mask, 'Background Mask'),
                (ax[1, 0], self.sample_mask, 'Sample Mask'),
                (ax[1, 1], J4M.line_mask * J4M.t_mask, 'Line and T Mask'),
                (ax[2, 0], self.poly_mask, 'Polygon Mask'),
            ]
            for a, data, title in mask_panels:
                self._plot_j4m_panel(a, data, title, fig, vmin=0, vmax=1)
            ax[2, 1].axis('off')
            plt.tight_layout()
            plt.show()

            # Combined mask
            fig, a = plt.subplots(figsize=(18, 14))
            self._plot_j4m_panel(a, self.cmask, 'Combined Mask', fig, vmin=0, vmax=1)
            plt.tight_layout()
            plt.show()

            # Sample average with combined mask
            fig, a = plt.subplots(figsize=(18, 14))
            self._plot_j4m_panel(a, sample_avg_combined, 'Sample Average with Combined Mask Applied', fig, vmin=0, cmap='jet')
            plt.tight_layout()
            plt.show()

        print(f'Total masked percentage: {100 * self._masked_fraction(self.cmask):.2f}%')

    def save_mask(
        self,
        valid_from_run: Optional[int] = None,
        mask_directory: Optional[str] = None,
    ) -> None:
        """Persist the combined mask to disk via *psana*.

        Parameters
        ----------
        valid_from_run : int, optional
            Run number from which the mask is considered valid.
            Defaults to :attr:`background_run_number`.
        mask_directory : str, optional
            Directory for the saved mask file.  Defaults to the standard
            LCLS calibration path for the experiment.
        """
        import psana  # lazy import — psana is only needed here

        if valid_from_run is None:
            valid_from_run = self.background_run_number
        if mask_directory is None:
            mask_directory = (
                f'/sdf/data/lcls/ds/cxi/{self.experiment}/'
                f'calib/Jungfrau::CalibV1/CxiDs1.0:Jungfrau.0/pixel_mask/'
            )

        os.makedirs(mask_directory, exist_ok=True)
        mask_filename = f'{valid_from_run}-end.data'
        mask_path = os.path.join(mask_directory, mask_filename)

        ds = psana.DataSource(f'exp={self.experiment}:run={self.dark_run_number}')
        det = psana.Detector('jungfrau4M')
        det.save_txtnda(mask_path, self.cmask.astype(float), fmt='%d', addmetad=True)
        print(f'Saved combined mask to {mask_path}')


# Backward-compatible alias for existing code
mask_maker = MaskMaker