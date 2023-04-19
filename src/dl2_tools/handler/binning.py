import numpy as np
from gammapy.maps import MapAxis
import astropy.units as u
from ctapipe.core import Component
from ctapipe.core.traits import List, Int

from .data_lists import (
    PointSourceSignalSetList,
)


class IRFBinning(Component):

    E_true_bins = List(
        default_value=[0.01 * u.TeV, 200 * u.TeV, 50],
        help="Emin, Emax and number of bins",
    ).tag(config=True)

    mig_bins = List(
        default_value=[0.01, 10, 50],
        help="Emin, Emax and number of bins",
    ).tag(config=True)

    psf_off_bins = List(
        default_value=[0.001 * u.deg, 2 * u.deg, 500],
        help="Emin, Emax and number of bins",
    ).tag(config=True)

    n_sig_offset_bins = Int(
        default_value=10,
        help="Number of signal offset bins if signal is diffuse",
    ).tag(config=True)

    n_bkg_offset_bins = Int(
        default_value=10,
        help="Number of background offset bins",
    ).tag(config=True)

    def __init__(self, observation, config=None, parent=None):

        super().__init__(config=config, parent=parent)

        self.energy_true = MapAxis(
            np.geomspace(self.E_true_bins[0], self.E_true_bins[1], self.E_true_bins[2]),
            interp="log",
            name="true_energy",
            node_type="edges",
        )
        self.migration = MapAxis(
            np.geomspace(self.mig_bins[0], self.mig_bins[1], self.mig_bins[2]),
            interp="log",
            name="migration",
            node_type="edges",
        )

        self.psf_offset = MapAxis(
            np.geomspace(
                self.psf_off_bins[0], self.psf_off_bins[1], self.psf_off_bins[2]
            ),
            interp="log",
            name="psf_offset",
            node_type="edges",
        )

        self.make_offset_bins_from_observation(
            observation,
            N_sig_bins=self.n_sig_offset_bins,
            N_bkg_bins=self.n_bkg_offset_bins,
        )

    def make_offset_bins_from_observation(
        self, observation, N_sig_bins=10, N_bkg_bins=10
    ):
        self._make_signal_offset_bins_from_observation(observation, N_sig_bins)
        if len(observation.background) != 0:
            self._make_background_offset_bins_from_observation(observation, N_bkg_bins)

    def _make_signal_offset_bins_from_observation(self, observation, N_sig_bins):

        if isinstance(observation.signal, PointSourceSignalSetList):

            self.signal_offset = self.make_ps_signal_offset_bins(observation.signal)

        else:

            self.signal_offset = self.make_diffuse_signal_offset_bins(
                observation.signal, N_sig_bins
            )

    @staticmethod
    def make_diffuse_signal_offset_bins(diffuse_signal_list, N_sig_bins):
        radii = diffuse_signal_list.get_radii()

        max_offset = np.max(radii)

        signal_offset = MapAxis(
            nodes=np.linspace[0.0, max_offset, N_sig_bins],
            name="signal_offset",
            node_type="edges",
            unit="deg",
        )
        return signal_offset

    @staticmethod
    def make_ps_signal_offset_bins(ps_signal_list):

        offsets = ps_signal_list.get_offsets()

        if offsets[0] == 0.0 * u.deg:

            if len(offsets) > 2:

                signal_offset_dummy = MapAxis(
                    nodes=offsets[1:],
                    node_type="center",
                    unit="deg",
                )
                signal_offset_high = MapAxis(
                    nodes=signal_offset_dummy.edges[1:],
                    node_type="edges",
                    unit="deg",
                )
                signal_offset_low = MapAxis(
                    nodes=[0.0, np.mean(offsets[:2])],
                    node_type="edges",
                    unit="deg",
                )
                signal_offset = MapAxis.from_stack(
                    [signal_offset_low, signal_offset_high]
                )

            elif len(offsets) == 1:
                signal_offset = MapAxis.from_edges(
                    u.Quantity([0.0 * u.deg, 0.1 * u.deg])
                )

            elif len(offsets) == 2:
                signal_offset_dummy = MapAxis(
                    nodes=offsets[1:],
                    node_type="center",
                    unit="deg",
                )

                signal_offset = MapAxis(
                    nodes=[
                        0.0,
                        np.mean(offsets[:2]),
                        signal_offset_dummy.edges[-1],
                    ],
                    node_type="edges",
                    unit="deg",
                )

        else:

            sig_offset = MapAxis(
                nodes=offsets,
                node_type="center",
                unit="deg",
            )

            if sig_offset.edges[0] < 0.0 * u.deg:

                signal_offset_high = MapAxis(
                    nodes=sig_offset.edges[2:],
                    node_type="edges",
                    unit="deg",
                )

                signal_offset_low = MapAxis(
                    nodes=[0.0, sig_offset.edges[1]],
                    node_type="edges",
                    unit="deg",
                )

                signal_offset = MapAxis.from_stack(
                    [signal_offset_low, signal_offset_high]
                )
        return signal_offset

    def _make_background_offset_bins_from_observation(self, observation, N_bkg_bins):

        radii = observation.background.get_radii()

        max_offset = np.max(radii)
        self.background_offset = MapAxis(
            nodes=np.linspace(0.0, max_offset, N_bkg_bins),
            name="background_offset",
            node_type="edges",
            unit="deg",
        )
