from ..handler.observation_handler import (
    PointSourceObservationHandler,
    DiffuseObservationHandler,
)
from ctapipe.core import Component
from ctapipe.core.traits import List, Int
from traitlets import Float
from ..handler.binning import IRFBinning
from .cut_optimizer import PercentileCutCalculator
import numpy as np
import astropy.units as u
from copy import deepcopy
from astropy.coordinates.angle_utilities import angular_separation
from gammapy.maps import MapAxis
from ..handler.interpolated_cuts import InterpolatedCut
import operator


class ROCMaker(Component):

    loss_params = List(
        default_value=[0.01, 0.99, 99],
        minlen=3,
        maxlen=3,
        help="min, max and number of bins",
    ).tag(config=True)

    radius_percentile = Float(
        help="Percentile of events to keep after radial cut for pointsource observation",
        default_value=68,
    ).tag(config=True)

    n_reco_energy_bins = Int(
        help="Number of energy bins to evaluate cut in ",
        default_value=10,
    ).tag(config=True)

    def __init__(self, config=None, parent=None):
        super().__init__(config=config, parent=parent)
        self.loss_vector = np.linspace(
            self.loss_params[0],
            self.loss_params[1],
            self.loss_params[2],
        )

    def __call__(self, observation):

        assert len(observation.background) >= 1

        self.binning = IRFBinning(observation, parent=self)

        if self.n_reco_energy_bins > 1:
            reco_energy_axis = [
                MapAxis(
                    nodes=np.geomspace(
                        self.binning.energy_true.edges[0],
                        self.binning.energy_true.edges[-1],
                        self.n_reco_energy_bins,
                    ),
                    node_type="edges",
                    name="reco_energy",
                )
            ]

        else:
            reco_energy_axis = None

        self.gh_cut_calculator = PercentileCutCalculator()
        self.gh_cut_calculator.cut_variable = "gh_score"
        self.gh_cut_calculator.op = operator.ge
        self.gh_cut_calculator.fill_value = 1
        self.gh_cut_calculator.min_events = 1
        self.gh_cut_calculator.bin_axes = reco_energy_axis

        if isinstance(observation, PointSourceObservationHandler):
            self.radius_cut_calculator = PercentileCutCalculator()
            self.radius_cut_calculator.cut_variable = "theta"
            self.radius_cut_calculator.op = operator.le
            self.radius_cut_calculator.cut_op = lambda x: x.to_value(u.rad)
            self.radius_cut_calculator.fill_value = np.deg2rad(0.1)
            self.radius_cut_calculator.percentile = self.radius_percentile
            self.radius_cut_calculator.bin_axes = reco_energy_axis

            self.purities = self._calc_ps_roc(
                observation,
            )

        elif isinstance(observation, DiffuseObservationHandler):

            self.purities = self._calc_ps_diff(
                observation,
            )

        else:
            raise TypeError(
                "observation must be either a PointSourceObservationhandler or a DiffuseObservationHandler"
            )

    def _calc_ps_roc(self, observation):
        purities = np.empty((len(self.loss_vector), len(observation.signal)))
        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            gh_cut_observation = deepcopy(observation)

            gh_cut_observation.add_cuts([gh_cut], "reco_source_fov_offset")

            radius_cut = self.radius_cut_calculator(
                gh_cut_observation.signal, "reco_source_fov_offset"
            )

            rate_observation = deepcopy(observation)
            rate_observation.add_signal_cuts(
                [gh_cut, radius_cut], "reco_source_fov_offset"
            )
            rate_observation.add_background_cuts([gh_cut], "reco_source_fov_offset")
            for j, signal in enumerate(rate_observation.signal):
                sig_evt_rate = np.sum(signal.get_masked_events()["weight"])
                bkg_evt_rate = 0
                for background in rate_observation.background:
                    rate_background = deepcopy(background)
                    rate_background.events["theta"] = angular_separation(
                        signal.source_position.az,
                        signal.source_position.alt,
                        rate_background.events["reco_az"],
                        rate_background.events["reco_alt"],
                    )

                    rate_background.add_cut(radius_cut, "reco_source_fov_offset")
                    bkg_evt_rate += np.sum(
                        rate_background.get_masked_events()["weight"]
                    )
                purities[i, j] = sig_evt_rate / (sig_evt_rate + bkg_evt_rate)

        return purities

    def _calc_roc_diff(self, observation):
        diff_offset_axis = self.binning.signal_offset
        purities = np.empty((len(self.loss_vector), len(diff_offset_axis.center)))

        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            gh_cut_observation = deepcopy(observation)

            gh_cut_observation.add_cuts([gh_cut], "reco_source_fov_offset")

            for j, (offset_edge_low, offset_edge_high) in enumerate(
                zip(diff_offset_axis.edges[:-1], diff_offset_axis.edges[1:])
            ):
                low_radius_cut = InterpolatedCut(
                    offset_axis=MapAxis(
                        nodes=[0.0 * u.deg],
                        node_type="center",
                        name="reco_source_fov_offset",
                    ),
                    cut_column="reco_source_fov_offset",
                    cut_values=[offset_edge_low],
                    method="nearest",
                    bounds_error=False,
                    fill_value=None,
                )

                high_radius_cut = InterpolatedCut(
                    offset_axis=MapAxis(
                        nodes=[-0.01 * u.deg, 0.01 * u.deg],
                        node_type="edges",
                        name="reco_source_fov_offset",
                    ),
                    cut_column="reco_source_fov_offset",
                    cut_values=[offset_edge_high],
                    op=operator.le,
                    method="nearest",
                    bounds_error=False,
                    fill_value=None,
                )

                rad_observation = deepcopy(gh_cut_observation)
                rad_observation.add_cuts(
                    [low_radius_cut, high_radius_cut], "reco_source_fov_offset"
                )
                bkg_evt_rate = 0
                for bkg in rad_observation.background:
                    bkg_evt_rate += np.sum(bkg.get_masked_events()["weight"])
                sig_evt_rate = np.sum(
                    rad_observation.signal[0].get_masked_events()["weight"]
                )
                purities[i, j] = sig_evt_rate / (sig_evt_rate + bkg_evt_rate)

        return purities

    def make_plot(self, ax):

        for j, cen in enumerate(self.binning.signal_offset.center):

            ax.plot(self.loss_vector, self.purities[:, j], label="{}".format(cen))

            ax.set_xlabel("Signal loss")
            ax.set_ylabel("Purity")
            ax.legend()

    def integrate_auc(self):
        aucs = np.empty(len(self.binning.signal_offset.center))
        for j in range(len(self.binning.signal_offset.center)):
            aucs[j] = np.sum(
                (self.loss_vector[1] - self.loss_vector[0]) * self.purities[:, j]
            )

        return aucs

    def plot_auc(self, ax):
        aucs = self.integrate_aucs()
        ax.scatter(self.binning.signal_offset.center, aucs, marker="d")
        ax.set_xlabel("FoV offset")
        ax.set_ylabel("AUC")
