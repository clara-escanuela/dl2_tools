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
from pyirf.sensitivity import estimate_background
from pyirf.binning import calculate_bin_indices

from scipy.integrate import quad
from scipy.interpolate import interp1d


def table_group_by(data, value_key, bins):
    bin_index, valid = calculate_bin_indices(data[value_key], bins)
    by_bin = data[valid].group_by(bin_index[valid])
    return by_bin


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

    bkg_ring_size = Float(
        help="Radius of FoV offset ring to consider for background estimation in deg",
        default_value=1,
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

        if self.n_reco_energy_bins >= 1:
            self.reco_energy_axis = MapAxis(
                nodes=np.geomspace(
                    self.binning.energy_true.edges[0],
                    self.binning.energy_true.edges[-1],
                    self.n_reco_energy_bins + 1,
                ),
                node_type="edges",
                name="reco_energy",
            )

        else:
            raise ValueError("You need to have at least one energy bin")

        self.gh_cut_calculator = PercentileCutCalculator()
        self.gh_cut_calculator.cut_variable = "gh_score"
        self.gh_cut_calculator.op = operator.ge
        self.gh_cut_calculator.fill_value = 1
        self.gh_cut_calculator.min_events = 1
        self.gh_cut_calculator.bin_axes = [self.reco_energy_axis]

        if isinstance(observation, PointSourceObservationHandler):
            self.radius_cut_calculator = PercentileCutCalculator()
            self.radius_cut_calculator.cut_variable = "theta"
            self.radius_cut_calculator.op = operator.le
            self.radius_cut_calculator.cut_op = lambda x: x.to_value(u.deg)
            self.radius_cut_calculator.fill_value = 0.1
            self.radius_cut_calculator.percentile = self.radius_percentile
            self.radius_cut_calculator.bin_axes = [self.reco_energy_axis]

            self.signal_loss, self.bkg_efficiencies = self._calc_roc_ps(
                observation,
            )

        elif isinstance(observation, DiffuseObservationHandler):

            self.signal_loss, self.bkg_efficiencies = self._calc_roc_diff(
                observation,
            )

        else:
            raise TypeError(
                "observation must be either a PointSourceObservationhandler or a DiffuseObservationHandler"
            )

    def _calc_roc_ps(self, observation):

        roc_sig_loss = np.empty(
            (
                len(self.loss_vector),
                len(observation.signal),
                len(self.reco_energy_axis.center),
            )
        )
        roc_bkg_efficiencies = np.empty(
            (len(self.loss_vector), len(self.reco_energy_axis.center))
        )

        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            (
                roc_sig_loss[i],
                roc_bkg_efficiencies[i],
            ) = self.calc_efficiencies_ps(observation, gh_cut)
        if len(observation.signal) > 1:
            return roc_sig_loss, roc_bkg_efficiencies
        else:
            return (
                roc_sig_loss.reshape(
                    len(self.loss_vector),
                    len(self.reco_energy_axis.center),
                ),
                roc_bkg_efficiencies,
            )

    def _calc_roc_diff(self, observation):

        roc_sig_loss = np.empty(
            (
                len(self.loss_vector),
                len(self.reco_energy_axis.center),
            )
        )
        roc_bkg_efficiencies = np.empty(
            (len(self.loss_vector), len(self.reco_energy_axis.center))
        )

        diff_offset_axis = self.binning.signal_offset
        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            (
                roc_sig_loss[i],
                roc_bkg_efficiencies[i],
            ) = self.calc_efficiencies_diff(observation, gh_cut, diff_offset_axis)

        return roc_sig_loss, roc_bkg_efficiencies

    def calc_efficiencies_ps(self, observation, gh_cut):

        sig_efficiencies = np.empty(
            (len(observation.signal), len(self.reco_energy_axis.center))
        )
        bkg_evt_rate = np.zeros(len(self.reco_energy_axis.center))
        uncut_bkg_evt_rate = np.zeros(len(self.reco_energy_axis.center))

        gh_cut_observation = deepcopy(observation)
        gh_cut_observation.add_cuts([gh_cut], "reco_source_fov_offset")
        radius_cut = self.radius_cut_calculator(
            gh_cut_observation.signal, "reco_source_fov_offset"
        )

        rate_observation = deepcopy(observation)
        rate_observation.add_signal_cuts([gh_cut, radius_cut], "reco_source_fov_offset")
        rate_observation.add_background_cuts([gh_cut], "reco_source_fov_offset")

        rad_cut_observation = deepcopy(observation)
        rad_cut_observation.add_signal_cuts([radius_cut], "reco_source_fov_offset")

        for j, (signal, uncut_signal) in enumerate(
            zip(rate_observation.signal, rad_cut_observation.signal)
        ):

            signal_by_bin = table_group_by(
                signal.get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            uncut_signal_by_bin = table_group_by(
                uncut_signal.get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            assert len(signal_by_bin.groups) == len(uncut_signal_by_bin.groups)
            for i, signal_group, uncut_signal_group in zip(
                signal_by_bin.groups.keys,
                signal_by_bin.groups,
                uncut_signal_by_bin.groups,
            ):
                uncut_sig_rate = np.sum(uncut_signal_group["weight"])
                sig_rate = np.sum(signal_group["weight"])
                sig_efficiencies[j, i] = sig_rate / uncut_sig_rate

        for j, (background, uncut_background) in enumerate(
            zip(rate_observation.background, rad_cut_observation.background)
        ):

            bg_hist_table = estimate_background(
                events=background.get_masked_events(),
                reco_energy_bins=self.reco_energy_axis.edges,
                theta_cuts=radius_cut.to_cut_table(
                    offset=0.0,
                    bin_axis=self.reco_energy_axis,
                    cut_column_unit=u.deg,
                ),
                alpha=1,
                fov_offset_min=max(
                    0.0 * u.deg, signal.offset - self.bkg_ring_size * u.deg
                ),
                fov_offset_max=signal.offset + self.bkg_ring_size * u.deg,
            )
            uncut_bg_hist_table = estimate_background(
                events=uncut_background.get_masked_events(),
                reco_energy_bins=self.reco_energy_axis.edges,
                theta_cuts=radius_cut.to_cut_table(
                    offset=0.0,
                    bin_axis=self.reco_energy_axis,
                    cut_column_unit=u.deg,
                ),
                alpha=1,
                fov_offset_min=max(
                    0.0 * u.deg, signal.offset - self.bkg_ring_size * u.deg
                ),
                fov_offset_max=signal.offset + self.bkg_ring_size * u.deg,
            )
            bkg_evt_rate += bg_hist_table["n_weighted"]
            uncut_bkg_evt_rate += uncut_bg_hist_table["n_weighted"]

        bkg_efficiencies = bkg_evt_rate / uncut_bkg_evt_rate

        return 1 - sig_efficiencies, bkg_efficiencies

    def calc_efficiencies_diff(self, observation, gh_cut, diff_offset_axis):

        sig_efficiencies = np.empty(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        bkg_efficiencies = np.empty(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        bkg_evt_rate = np.zeros(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        uncut_bkg_evt_rate = np.zeros(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )

        assert isinstance(observation, DiffuseObservationHandler)

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
            rad_gh_observation = deepcopy(gh_cut_observation)
            rad_gh_observation.add_cuts(
                [low_radius_cut, high_radius_cut], "reco_source_fov_offset"
            )

            rad_observation = deepcopy(observation)
            rad_observation.add_cuts(
                [low_radius_cut, high_radius_cut], "reco_source_fov_offset"
            )

            signal_by_bin = table_group_by(
                rad_gh_observation.signal[0].get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            uncut_signal_by_bin = table_group_by(
                rad_observation.signal[0].get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            assert len(signal_by_bin.groups) == len(uncut_signal_by_bin.groups)
            for i, signal_group, uncut_signal_group in zip(
                signal_by_bin.groups.keys,
                signal_by_bin.groups,
                uncut_signal_by_bin.groups,
            ):
                uncut_sig_rate = np.sum(uncut_signal_group["weight"])
                sig_rate = np.sum(signal_group["weight"])
                sig_efficiencies[j, i] = sig_rate / uncut_sig_rate

            for bkg, uncut_bkg in zip(
                rad_gh_observation.background, rad_observation.background
            ):
                background_by_bin = table_group_by(
                    bkg.get_masked_events(),
                    "reco_energy",
                    self.reco_energy_axis.edges,
                )
                uncut_background_by_bin = table_group_by(
                    uncut_bkg.get_masked_events(),
                    "reco_energy",
                    self.reco_energy_axis.edges,
                )
                for i, bkg_group, uncut_bkg_group in zip(
                    background_by_bin.groups.keys,
                    background_by_bin.groups,
                    uncut_background_by_bin.groups,
                ):
                    uncut_bkg_evt_rate[j, i] += np.sum(uncut_bkg_group["weight"])
                    bkg_evt_rate[j, i] += np.sum(bkg_group["weight"])

        bkg_efficiencies = bkg_evt_rate / uncut_bkg_evt_rate

        return 1 - sig_efficiencies, bkg_efficiencies

    @staticmethod
    def make_plot(ax, signal_loss, bkg_efficiencies, **kwargs):

        sort_args = np.argsort(signal_loss)
        sorted_sig_effs = np.sort(signal_loss)
        sorted_bkg_effs = bkg_efficiencies[sort_args]

        ax.plot(sorted_sig_effs, sorted_bkg_effs, **kwargs)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Signal loss")
        ax.set_ylabel("Background efficiency")

    @staticmethod
    def make_energy_plot(ax, energies, bkg_efficiencies, **kwargs):

        ax.plot(energies, bkg_efficiencies, **kwargs)

        ax.set_ylabel("Background efficiency")
        ax.set_xlabel("Energy in TeV")
        ax.set_xscale("log")
        ax.set_yscale("log")

    @staticmethod
    def integrate_auc(signal_loss, bkg_efficiencies, **kwargs):

        sort_args = np.argsort(signal_loss)
        sorted_sig_effs = np.sort(signal_loss)
        sorted_bkg_effs = bkg_efficiencies[sort_args]

        f = lambda x: np.interp(
            x,
            sorted_sig_effs,
            sorted_bkg_effs,
        )
        auc = quad(f, 0, 1)

        return auc[0]

    def integrate_auc_all_energy_bins(self, offset):

        aucs = np.empty(len(self.reco_energy_axis.center))

        if len(np.shape(self.signal_loss)) > 2:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            for i, (sig_loss, bkg_eff) in enumerate(
                zip(
                    self.signal_loss[:, offset_bin_id, :].T,
                    self.bkg_efficiencies[:, offset_bin_id, :].T,
                )
            ):
                aucs[i] = ROCMaker.integrate_auc(sig_loss, bkg_eff)

        else:
            assert len(self.signal_loss.T[0]) == len(self.loss_vector)
            for i, (sig_loss, bkg_eff) in enumerate(
                zip(
                    self.signal_loss.T,
                    self.bkg_efficiencies.T,
                )
            ):
                aucs[i] = ROCMaker.integrate_auc(sig_loss, bkg_eff)

        return aucs

    def integrate_auc_single_energy(self, energy, offset):
        bin_id = self.reco_energy_axis.coord_to_idx(energy, clip=True)

        if len(np.shape(self.signal_loss)) == 2:
            auc = ROCMaker.integrate_auc(
                self.signal_loss.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            auc = ROCMaker.integrate_auc(
                self.signal_loss[:, offset_bin_id, :]
                .T[bin_id]
                .reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies[:, offset_bin_id, :]
                .T[bin_id]
                .reshape(
                    len(self.loss_vector),
                ),
            )

        return auc

    def roc_plot_all_energy_bins(self, ax, offset, **kwargs):

        if len(np.shape(self.signal_loss)) > 2:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            for E, sig_loss, bkg_eff in zip(
                self.reco_energy_axis.center,
                self.signal_loss[:, offset_bin_id, :].T,
                self.bkg_efficiencies[:, offset_bin_id, :].T,
            ):
                ROCMaker.make_plot(ax, sig_loss, bkg_eff, label=E, **kwargs)

        else:
            assert len(self.signal_loss.T[0]) == len(self.loss_vector)
            for E, sig_loss, bkg_eff in zip(
                self.reco_energy_axis.center,
                self.signal_loss.T,
                self.bkg_efficiencies.T,
            ):
                ROCMaker.make_plot(ax, sig_loss, bkg_eff, label=E, **kwargs)

        ax.legend()

    def roc_plot_single_energy(self, ax, energy, offset, **kwargs):
        bin_id = self.reco_energy_axis.coord_to_idx(energy, clip=True)

        if len(np.shape(self.signal_loss)) == 2:
            ROCMaker.make_plot(
                ax,
                self.signal_loss.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                label=energy,
                **kwargs
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            ROCMaker.make_plot(
                ax,
                self.signal_loss[:, offset_bin_id, :].T[bin_id],
                self.bkg_efficiencies[:, offset_bin_id, :].T[bin_id],
                label=energy,
                **kwargs
            )

    def energy_plot_efficiencies(self, ax, efficiency, offset, **kwargs):

        loss = 1 - efficiency

        assert loss > np.min(self.signal_loss)
        assert loss < np.max(self.signal_loss)

        if len(np.shape(self.signal_loss)) == 2:

            smaller_loss = np.max(
                np.where(self.signal_loss[:, 0] <= loss, self.signal_loss[:, 0], 0)
            )
            larger_loss = np.min(
                np.where(self.signal_loss[:, 0] > loss, self.signal_loss[:, 0], 1)
            )

            smaller_ind = np.argwhere(self.signal_loss[:, 0] == smaller_loss)
            larger_ind = np.argwhere(self.signal_loss[:, 0] == larger_loss)

            bkg_effs = (larger_loss - loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[smaller_ind] + (loss - smaller_loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[
                larger_ind
            ]

            ROCMaker.make_energy_plot(
                ax,
                self.reco_energy_axis.center,
                bkg_effs.reshape(
                    len(self.reco_energy_axis.center),
                ),
                label="Signal efficiency={}".format(efficiency),
                **kwargs
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            smaller_loss = np.max(
                np.where(
                    self.signal_loss[:, offset_bin_id, 0] <= loss,
                    self.signal_loss[:, offset_bin_id, 0],
                    0,
                )
            )
            larger_loss = np.min(
                np.where(
                    self.signal_loss[:, offset_bin_id, 0] > loss,
                    self.signal_loss[:, offset_bin_id, 0],
                    1,
                )
            )

            smaller_ind = np.argwhere(
                self.signal_loss[:, offset_bin_id, 0] == smaller_loss
            )
            larger_ind = np.argwhere(
                self.signal_loss[:, offset_bin_id, 0] == larger_loss
            )

            bkg_effs = (larger_loss - loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[smaller_ind, offset_bin_id] + (
                loss - smaller_loss
            ) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[
                larger_ind, offset_bin_id
            ]

            ROCMaker.make_energy_plot(
                ax,
                self.reco_energy_axis.center,
                bkg_effs.reshape(
                    len(self.reco_energy_axis.center),
                ),
                label="Signal efficiency={}".format(efficiency),
                **kwargs
            )

        ax.legend()
