from abc import ABC, abstractmethod
import numpy as np
from pyirf.spectral import calculate_event_weights, PowerLaw
from pyirf.simulations import SimulatedEventsInfo
from astropy.table import QTable
from collections import Counter
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import AltAz
import astropy.units as u
from ctapipe.io import TableLoader
from copy import copy


class SimDataset(ABC):
    """
    Base class for handling dl2 datasets
    """

    def __init__(self, loader, reco_energy_name, gh_score_name, obs_time=1 * u.s):
        """
        Constructor of the Base Class. Initializes Datset from ctapipe TableLoader

        Parameters
        ----------
        loader : ctapipie.io.TableLoader
            ctapipe TableLoader object conatining dl2 data
        reco_energy_name : string
            Prefix of the reconstructed energy columns in the table
        gh_score_name : string
            Prefix of the g/h separation score columns
        obs_time : u.Quantity time, optional
            Observation time of the dataset. Needed to make the weights.
            By default 1*u.s, this makes it easy to calculate weights in 1/s
        """
        self.obs_time = obs_time
        self.events = QTable(loader.read_subarray_events())

        self._set_reco_energy_column(reco_energy_name)
        self._set_gh_score_column(gh_score_name)

    @abstractmethod
    def _set_simulation_info(self, sim_config, obs_info):
        """Reads metadata from the TableLoader and creates SimulationInfo dictionary.

        Parameters
        ----------
        sim_config : astropy.table.QTable
            Created from TableLoader using read_simulation_configuration()
        obs_info : astropy.table.QTable
            Created from TableLoader using read_observation_information()
        """

        # The rows of sim_config and obs_info list the different simualtion files that went into them.
        # The asserts below make sure that certain items on the metadata are the same between them.
        assert (
            len(Counter(sim_config["energy_range_min"]).keys()) == 1
        ), "Minimum energies do not match between simulations"
        assert (
            len(Counter(sim_config["energy_range_max"]).keys()) == 1
        ), "Maximum energies do not match between simulations"
        assert (
            len(Counter(sim_config["spectral_index"]).keys()) == 1
        ), "Spectral indicies do not match between simulations"
        assert (
            len(Counter(sim_config["max_viewcone_radius"]).keys()) == 1
        ), "Viewcones do not match between simulations"
        assert (
            len(Counter(sim_config["max_scatter_range"]).keys()) == 1
        ), "Maximum impact distances do not match between simulations"
        assert (
            len(Counter(sim_config["diffuse"]).keys()) == 1
        ), "Diffuse and non-diffuse simulations are mixed here"
        assert (
            len(Counter(obs_info["subarray_pointing_lon"]).keys()) == 1
        ), "There are different pointing directions among the simulation"
        assert (
            len(Counter(obs_info["subarray_pointing_lat"]).keys()) == 1
        ), "There are different pointing directions among the simulation"
        assert (
            len(Counter(obs_info["subarray_pointing_lat"]).keys()) == 1
        ), "There are different pointing directions among the simulation"

        # If we later want to compare observations at different zeniths, we better make
        # the pointing direction a class attribute
        self.pointing_direction = AltAz(
            az=obs_info["subarray_pointing_lon"][0],
            alt=obs_info["subarray_pointing_lat"][0],
        )

    def _set_simulation_weights(self):
        """
        Generates columns "weight" and "weight_rate" based on the simulated
        energy spectrum and specified observation time
        """
        self.spectrum = PowerLaw.from_simulation(self.simulation_info, self.obs_time)
        self.events["weight"] = np.ones(len(self.events))
        self.events["weight_rate"] = np.repeat(
            u.Quantity(1 / self.obs_time), len(self.events)
        )

    def _set_reco_energy_column(self, reco_energy_name):
        reco_energy_string = f"{reco_energy_name}_energy"
        if reco_energy_name != "ImPACTReconstructor":
            reco_energy_valid_string = f"{reco_energy_name}_is_valid"
        else:
            reco_energy_valid_string = f"{reco_energy_name}_is_valid_1"
        self.events["reco_energy"] = self.events[reco_energy_string]
        self.events["reco_energy_is_valid"] = self.events[reco_energy_valid_string]

    def _set_gh_score_column(self, gh_score_name):
        gh_score_string = f"{gh_score_name}_prediction"
        gh_score_valid_string = f"{gh_score_name}_is_valid"
        self.events["gh_score"] = self.events[gh_score_string]
        self.events["gh_score_is_valid"] = self.events[gh_score_valid_string]

    @abstractmethod
    def _set_geometry_columns(self, geometry_reco_name):
        self.events["reco_az"] = self.events[f"{geometry_reco_name}_az"]
        self.events["reco_alt"] = self.events[f"{geometry_reco_name}_alt"]
        if geometry_reco_name != "ImPACTReconstructor":
            self.events["geometry_reco_is_valid"] = self.events[
                f"{geometry_reco_name}_is_valid"
            ]
        else:
            self.events["geometry_reco_is_valid"] = self.events[
                f"{geometry_reco_name}_is_valid_1"
            ]

        self.events["reco_source_fov_offset"] = angular_separation(
            self.pointing_direction.az,
            self.pointing_direction.alt,
            self.events["reco_az"],
            self.events["reco_alt"],
        )

    def _set_valid_column(self):
        self.events["reco_is_valid"] = np.logical_and(
            self.events["geometry_reco_is_valid"],
            np.logical_and(
                self.events["gh_score_is_valid"], self.events["reco_energy_is_valid"]
            ),
        )

    @classmethod
    def from_path(cls, path, reco_energy_name, gh_score_name, obs_time=1 * u.s):
        loader = TableLoader(
            path,
            load_true_images=True,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_simulated=True,
            load_dl2=True,
        )
        return cls(loader, reco_energy_name, gh_score_name, obs_time)

    def reweight_to(self, new_spectrum):
        reweight_factors = calculate_event_weights(
            self.events["true_energy"], new_spectrum, self.spectrum
        )
        self.events["weight"] = self.events["weight"] * reweight_factors
        self.events["weight_rate"] = self.events["weight_rate"] * reweight_factors
        self.spectrum = new_spectrum

    def set_obs_time(self, obs_time):
        self.events["weight"] = (self.events["weight_rate"] * obs_time).to_value(u.one)
        self.obs_time = obs_time

    def set_cuts(self, cuts, offset_column=None):

        cuts_mask = np.ones(len(self.events))
        self.events["passes_cuts"] = cuts_mask
        if cuts is not None:
            assert offset_column is not None, "You need to specify an offset column"
            for cut in cuts:
                self.add_cut(cut, offset_column)

    def add_cut(self, cut, offset_column):

        assert (
            cut.cut_column in self.events.keys()
        ), "The chosen cut parameter is not in the events table"

        if cut.bin_columns is not None:
            interp_cols = copy(cut.bin_columns)
            interp_cols.insert(0, offset_column)
        else:
            interp_cols = [offset_column]
        interp_array = []
        for col in interp_cols:
            if u.Quantity(self.events[col]).unit.is_equivalent(u.TeV):
                interp_array.append(np.log10(self.events[col].to_value(u.TeV)))
            else:
                interp_array.append(self.events[col])
        interp_array = np.array(interp_array)
        if not cut.log_cut:
            cut_values = cut.cut_spline(interp_array.T)
        else:
            cut_values = 10 ** cut.cut_spline(interp_array.T)
        cuts_mask = np.logical_and(
            cut.op(cut.cut_op(self.events[cut.cut_column]), cut_values),
            self.events["passes_cuts"],
        )

        self.events["passes_cuts"] = cuts_mask

    def get_masked_events(self):
        mask = np.logical_and(self.events["passes_cuts"], self.events["reco_is_valid"])
        return self.events[mask]


class DiffuseDataset(SimDataset):
    def __init__(self, loader, reco_energy_name, gh_score_name, obs_time=1 * u.s):
        super().__init__(loader, reco_energy_name, gh_score_name, obs_time)
        sim_config = QTable(loader.read_simulation_configuration())
        obs_info = QTable(loader.read_observation_information())
        self._set_simulation_info(sim_config, obs_info)
        self._set_simulation_weights()

    def _set_simulation_info(self, sim_config, obs_info):
        super()._set_simulation_info(sim_config, obs_info)
        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        assert sim_config["diffuse"][
            0
        ], "A diffuse dataset must be made from diffuse simulation"

        self.simulation_info = SimulatedEventsInfo(
            n_showers=np.sum(sim_config["n_showers"] * sim_config["shower_reuse"]),
            max_impact=sim_config["max_scatter_range"][0],
            viewcone=sim_config["max_viewcone_radius"][0],
            energy_min=sim_config["energy_range_min"][0],
            energy_max=sim_config["energy_range_max"][0],
            spectral_index=sim_config["spectral_index"][0],
        )

        self.source_position = AltAz(
            az=obs_info["subarray_pointing_lon"][0],
            alt=obs_info["subarray_pointing_lat"][0],
        )
        self.radius = self.simulation_info.viewcone


class PointSourceDataset(SimDataset):
    def __init__(self, loader, reco_energy_name, gh_score_name, obs_time=1 * u.s):
        super().__init__(loader, reco_energy_name, gh_score_name, obs_time)
        sim_config = QTable(loader.read_simulation_configuration())
        obs_info = QTable(loader.read_observation_information())
        self._set_simulation_info(sim_config, obs_info)
        self._set_simulation_weights()

    def _set_simulation_info(self, sim_config, obs_info):
        super()._set_simulation_info(sim_config, obs_info)
        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        assert not sim_config["diffuse"][
            0
        ], "A point source dataset must be made from point source simulation"

        offsets = (
            angular_separation(
                obs_info["subarray_pointing_lon"],
                obs_info["subarray_pointing_lat"],
                sim_config["max_az"],
                sim_config["max_alt"],
            )
            .to_value(u.deg)
            .round(1)
        )  # If we don't round we get issues with floating precision.
        # Also, if offsets are that close they may as well be the same

        assert (
            len(set(offsets)) == 1
        ), "There are different offsets among the simulations"

        self.offset = offsets[0] * u.deg
        self.source_position = AltAz(
            az=sim_config["max_az"][0], alt=sim_config["max_alt"][0]
        )

        self.simulation_info = SimulatedEventsInfo(
            n_showers=np.sum(sim_config["n_showers"] * sim_config["shower_reuse"]),
            max_impact=sim_config["max_scatter_range"][0],
            viewcone=sim_config["max_viewcone_radius"][0],
            energy_min=sim_config["energy_range_min"][0],
            energy_max=sim_config["energy_range_max"][0],
            spectral_index=sim_config["spectral_index"][0],
        )


class SignalSet(SimDataset):
    def __init__(
        self,
        loader,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name,
        obs_time=1 * u.s,
    ):
        super().__init__(loader, reco_energy_name, gh_score_name, obs_time)
        self._set_geometry_columns(geometry_reco_name)
        self._set_valid_column()
        self.set_cuts(None)

    def _set_geometry_columns(self, geometry_reco_name):
        super()._set_geometry_columns(geometry_reco_name)
        self.events["true_source_fov_offset"] = angular_separation(
            self.pointing_direction.az,
            self.pointing_direction.alt,
            self.events["true_az"],
            self.events["true_alt"],
        )

        self.events["theta"] = angular_separation(
            self.events["reco_az"],
            self.events["reco_alt"],
            self.events["true_az"],
            self.events["true_alt"],
        )

    @classmethod
    def from_path(
        cls, path, reco_energy_name, geometry_name, gh_score_name, obs_time=1 * u.s
    ):
        loader = TableLoader(
            path,
            load_true_images=True,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_simulated=True,
            load_dl2=True,
        )
        return cls(loader, reco_energy_name, geometry_name, gh_score_name, obs_time)


class PointSourceSignalSet(SignalSet, PointSourceDataset):
    def __init__(
        self,
        loader,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name,
        obs_time=1 * u.s,
    ):
        super().__init__(
            loader, reco_energy_name, geometry_reco_name, gh_score_name, obs_time
        )


class DiffuseSignalSet(SignalSet, DiffuseDataset):
    def __init__(
        self,
        loader,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name,
        obs_time=1 * u.s,
    ):
        super().__init__(
            loader, reco_energy_name, geometry_reco_name, gh_score_name, obs_time
        )


class BackgroundSet(DiffuseDataset):
    def __init__(
        self,
        loader,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name,
        obs_time=1 * u.s,
    ):
        super().__init__(loader, reco_energy_name, gh_score_name, obs_time)
        self._set_geometry_columns(geometry_reco_name)
        self._set_valid_column()
        self.set_cuts(None)

    def _set_geometry_columns(self, geometry_reco_name):
        super()._set_geometry_columns(geometry_reco_name)

    @classmethod
    def from_path(
        cls, path, reco_energy_name, geometry_name, gh_score_name, obs_time=1 * u.s
    ):
        loader = TableLoader(
            path,
            load_true_images=True,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_simulated=True,
            load_dl2=True,
        )
        return cls(loader, reco_energy_name, geometry_name, gh_score_name, obs_time)
