# This should take an observation handler after gamma hadron cuts and calculate all the IRFS from it
# These can then be passed to the cut optimizer for gammapy cut optimization or saved to disk
# It needs to cnatin conditions on the Observationhandler in order to produce irfs from it

from ..handler.irf_handler import IRFHandler
from pyirf.irf import (
    energy_dispersion,
    effective_area_per_energy,
    effective_area_per_energy_and_fov,
    psf_table,
    background_2d,
    BACKGROUND_UNIT,
)
from pyirf.gammapy import (
    create_effective_area_table_2d,
    create_energy_dispersion_2d,
    create_psf_3d,
    create_background_2d,
)

from pyirf.benchmarks import energy_bias_resolution_from_energy_dispersion
import astropy.units as u
import numpy as np

from ..handler.data_lists import DiffuseSignalSetList

from ctapipe.core import Component

from scipy.interpolate import RegularGridInterpolator

from ..handler.binning import IRFBinning


class IRFMaker(Component):
    def __init__(self, config=None, parent=None) -> None:
        super().__init__(config=config, parent=parent)

    def __call__(self, observation, binning=None):
        if binning is None:
            self.binning = IRFBinning(observation, parent=self)
        else:
            self.binning = binning
        # Check conditions on observations

        irfs = IRFHandler()

        if len(observation.signal) > 0:
            irfs.edisp, irfs.energy_bias, irfs.energy_resolution = self._make_Edisps(
                observation.signal
            )
            irfs.aeff = self._make_Aeff(observation.signal)
            irfs.psf = self._make_psf(observation.signal)
            irfs.binning = self.binning

        if len(observation.background) != 0:
            irfs.bkg_model = self._make_background_model(observation.background)

        return irfs

    def _make_background_model(self, background):
        pyirf_bkg = u.Quantity(
            np.zeros(
                (
                    len(self.binning.energy_reco.center),
                    len(
                        self.binning.background_offset.center,
                    ),
                )
            )
            * BACKGROUND_UNIT
        )

        for bkg in background:
            single_bkg = background_2d(
                bkg.get_masked_events(),
                self.binning.energy_reco.edges,
                self.binning.background_offset.edges,
                bkg.obs_time,
            )
            pyirf_bkg += single_bkg

        gammapy_bkg = create_background_2d(
            pyirf_bkg,
            self.binning.energy_reco.edges,
            self.binning.background_offset.edges,
        )

        return gammapy_bkg

    def _make_Edisps(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            edisp_pyirf = self._make_Edisps_diffuse(signal[0])
            offset_points = self.binning.signal_offset.center.to_value(u.deg)

        else:
            edisp_pyirf = self._make_Edisps_pl(signal)
            offset_points = signal.get_offsets().to_value(u.deg)

        edisp = create_energy_dispersion_2d(
            edisp_pyirf,
            self.binning.energy_true.edges,
            self.binning.migration.edges,
            self.binning.signal_offset.edges,
        )

        (
            pyirf_energy_bias,
            pyirf_energy_resolution,
        ) = energy_bias_resolution_from_energy_dispersion(
            edisp_pyirf, self.binning.migration.edges
        )

        energy_bias_spline = RegularGridInterpolator(
            (
                np.log10(self.binning.energy_true.center.to_value(u.TeV)),
                offset_points,
            ),
            pyirf_energy_bias,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        energy_resolution_spline = RegularGridInterpolator(
            (
                np.log10(self.binning.energy_true.center.to_value(u.TeV)),
                offset_points,
            ),
            pyirf_energy_resolution,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        return edisp, energy_bias_spline, energy_resolution_spline

    def _make_Edisps_diffuse(self, signal):
        edisp_pyirf = energy_dispersion(
            signal.get_masked_events(),
            true_energy_bins=self.binning.energy_true.edges,
            migration_bins=self.binning.migration.edges,
            fov_offset_bins=self.binning.signal_offset.edges,
        )
        return edisp_pyirf

    def _make_Edisps_pl(self, signal):
        edisp_pyirf = np.empty(
            (
                len(self.binning.energy_true.center),
                len(self.binning.migration.center),
                len(self.binning.signal_offset.center),
            )
        )

        for i, _sig in enumerate(signal):
            edisp_pyirf[:, :, i] = energy_dispersion(
                _sig.get_masked_events(),
                true_energy_bins=self.binning.energy_true.edges,
                migration_bins=self.binning.migration.edges,
                fov_offset_bins=u.Quantity(
                    [
                        self.binning.signal_offset.edges[i],
                        self.binning.signal_offset.edges[i + 1],
                    ]
                ),
                use_event_weights=True
            ).reshape(
                len(self.binning.energy_true.center), len(self.binning.migration.center)
            )

        return edisp_pyirf

    def _make_Aeff(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            aeff_pyirf = self._make_Aeff_diffuse(signal[0])
        else:
            aeff_pyirf = self._make_Aeff_pl(signal)

        aeff = create_effective_area_table_2d(
            aeff_pyirf, self.binning.energy_true.edges, self.binning.signal_offset.edges
        )
        return aeff

    def _make_Aeff_pl(self, signal):
        aeff_pyirf = u.Quantity(
            np.empty(
                (
                    len(self.binning.energy_true.center),
                    len(self.binning.signal_offset.center),
                )
            ),
            u.m**2,
        )
        for i, _sig in enumerate(signal):
            aeff_pyirf[:, i] = effective_area_per_energy(
                _sig.get_masked_events(),
                _sig.simulation_info,
                self.binning.energy_true.edges,
            )

        return aeff_pyirf

    def _make_Aeff_diffuse(self, signal):
        aeff_pyirf = effective_area_per_energy_and_fov(
            signal.get_masked_events(),
            signal.simulation_info,
            self.binning.energy_true.edges,
            self.binning.signal_offset.edges,
        )
        return aeff_pyirf

    def _make_psf(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            psf_pyirf = self._make_psf_diffuse(signal[0])
        else:
            psf_pyirf = self._make_psf_pl(signal)

        psf = create_psf_3d(
            psf_pyirf,
            self.binning.energy_true.edges,
            self.binning.psf_offset.edges,
            self.binning.signal_offset.edges,
        )
        return psf

    def _make_psf_pl(self, signal):
        psf_pyirf = u.Quantity(
            np.empty(
                (
                    len(self.binning.energy_true.center),
                    len(self.binning.signal_offset.center),
                    len(self.binning.psf_offset.center),
                )
            ),
            unit=u.sr**-1,
        )

        for i, _sig in enumerate(signal):
            psf_pyirf[:, i, :] = np.swapaxes(
                psf_table(
                    _sig.get_masked_events(),
                    true_energy_bins=self.binning.energy_true.edges,
                    source_offset_bins=self.binning.psf_offset.edges,
                    fov_offset_bins=u.Quantity(
                        [
                            self.binning.signal_offset.edges[i],
                            self.binning.signal_offset.edges[i + 1],
                        ]
                    ),
                    use_event_weights=True
                ),
                1,
                2,
            ).reshape(
                len(self.binning.energy_true.center),
                len(self.binning.psf_offset.center),
            )
        return psf_pyirf

    def _make_psf_diffuse(self, signal):
        psf_pyirf = psf_table(
            signal.get_masked_events(),
            true_energy_bins=self.binning.energy_true.edges,
            source_offset_bins=self.binning.psf_offset.edges,
            fov_offset_bins=self.binning.signal_offset.edges,
        )

        return psf_pyirf
