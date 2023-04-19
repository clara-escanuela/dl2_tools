from .data_lists import (
    PointSourceSignalSetList,
    DiffuseSignalSetList,
    BackgroundSetList,
    DataList,
)

from abc import ABC
import numpy as np
import astropy.units as u


class ObservationHandler(ABC):
    def __init__(self, signal_list=None, background_list=None) -> None:

        assert isinstance(signal_list, DataList)
        assert isinstance(background_list, BackgroundSetList)
        self.signal = signal_list
        self.background = background_list

        pointing_directions = [
            self.signal.pointing_direction,
            self.background.pointing_direction,
        ]

        if all(dir is None for dir in pointing_directions):
            self.pointing_direction = None
        else:
            pointing_directions = [
                dir for dir in pointing_directions if dir is not None
            ]

            pointing_direction_floats = [
                1000 * dir.alt.to_value(u.deg) + dir.az.to_value(u.deg)
                for dir in pointing_directions
                if dir is not None
            ]

            assert (
                len(np.unique(pointing_direction_floats)) == 1
            ), "Signals and background need to be from observations pointing in the same direction"
            self.pointing_direction = pointing_directions[0]

    def add_signal(self, signal):
        if self.pointing_direction is None:
            self.pointing_direction = signal.pointing_direction
        else:
            assert self.pointing_direction == signal.pointing_direction

        self.signal.append(signal)

    def add_background(self, background):
        if self.pointing_direction is None:
            self.pointing_direction = background.pointing_direction
        else:
            assert self.pointing_direction == background.pointing_direction

        self.background.append(background)

    def set_obs_time_all(self, obs_time):
        self.signal.set_obs_time(obs_time)
        self.background.set_obs_time(obs_time)

    def set_signal_cuts(self, cuts, offset_column):
        self.signal.set_cuts(cuts, offset_column)

    def set_background_cuts(self, cuts, offset_column):
        self.background.set_cuts(cuts, offset_column)

    def set_cuts(
        self, signal_cuts, background_cuts, sig_offset_column, bkg_offset_column
    ):

        self.set_signal_cuts(signal_cuts, sig_offset_column)
        self.set_background_cuts(background_cuts, bkg_offset_column)

    def add_signal_cuts(self, cuts, offset_column):
        for cut in cuts:
            self.signal.add_cut(cut, offset_column)

    def add_background_cuts(self, cuts, offset_column):
        for cut in cuts:
            self.background.add_cut(cut, offset_column)

    def add_cuts(self, cuts, offset_column):
        self.add_signal_cuts(cuts, offset_column)
        self.add_background_cuts(cuts, offset_column)


class PointSourceObservationHandler(ObservationHandler):
    def __init__(self, signal=None, background=None):

        background_list = BackgroundSetList(data=background)

        signal_list = PointSourceSignalSetList(data=signal)

        super().__init__(signal_list, background_list)


class DiffuseObservationHandler(ObservationHandler):
    def __init__(self, signal=None, background=None):

        background_list = BackgroundSetList(data=background)

        signal_list = DiffuseSignalSetList(data=signal)

        super().__init__(signal_list, background_list)
