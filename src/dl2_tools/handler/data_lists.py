from collections.abc import MutableSequence
from .sim_datasets import (
    PointSourceDataset,
    DiffuseDataset,
    PointSourceSignalSet,
    DiffuseSignalSet,
    BackgroundSet,
)
import astropy.units as u


class DataList(MutableSequence):

    """
    See https://stackoverflow.com/questions/3487434/overriding-append-method-after-inheriting-from-a-python-list/3488283#3488283
    This is a container for multiple (simulation) datasets at different FoV offsets
    """

    def __init__(self, type, data=None):
        """
        Initialize the class.
        type specifies the type for the object contained in the list.
        """
        super(DataList, self).__init__()
        self.type = type
        if data is not None:
            self._list = list(data)
            self.pointing_direction = self._list[0].pointing_direction
            for entry in self._list:
                self.check(entry)
                assert entry.pointing_direction == self.pointing_direction
        else:
            self._list = list()
            self.pointing_direction = None

    def __len__(self):
        """List length"""
        return len(self._list)

    def __getitem__(self, ii):
        """Get a list item"""
        if isinstance(ii, slice):
            return self.__class__(self._list[ii])
        else:
            return self._list[ii]

    def __delitem__(self, ii):
        """Delete an item"""
        del self._list[ii]

    def check(self, v):
        "Check if an object is of the type stored in the list"
        if not isinstance(v, self.type):
            raise TypeError(v)

    def __setitem__(self, i, v):
        self.check(v)

        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        self._list[i] = v

    def insert(self, i, v):
        self.check(v)
        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        self._list.insert(i, v)

    def reweight_to(self, new_spectrum):
        for entry in self._list:
            entry.reweight_to(new_spectrum)

    def set_obs_time(self, obs_time):
        for entry in self._list:
            entry.set_obs_time(obs_time)

    def set_cuts(self, cuts, offset_column):
        for entry in self._list:
            entry.set_cuts(cuts, offset_column)

    def add_cut(self, cut, offset_column):
        for entry in self._list:
            entry.add_cut(cut, offset_column)


class PointSourceSetList(DataList):
    def __init__(self, type, data=None):
        assert issubclass(type, PointSourceDataset)
        super().__init__(type, data)

        self._list.sort(key=lambda e: e.offset)
        if data is not None:
            assert [
                x
                for x in list(self.get_offsets().to_value(u.deg))
                if list(self.get_offsets().to_value(u.deg)).count(x) == 1
            ] == self.get_offsets(), "Any offset can appear in the list only once"

    def __setitem__(self, i, v):
        assert v.offset not in self.get_offsets()
        super().__setitem__(i, v)
        self._list.sort(key=lambda e: e.offset)

    def insert(self, i, v):
        assert v.offset not in self.get_offsets()
        super().insert(i, v)
        self._list.sort(key=lambda e: e.offset)

    def get_offsets(self):
        return u.Quantity([entry.offset for entry in self._list])

    def get_source_positions(self):
        return [entry.source_position for entry in self._list]


class DiffuseSetList(DataList):
    def __init__(self, type, data=None):
        assert issubclass(type, DiffuseDataset)
        super().__init__(type, data)

    def get_radii(self):
        return u.Quantity([entry.radius for entry in self._list])


class PointSourceSignalSetList(PointSourceSetList):
    def __init__(self, data=None):
        super().__init__(PointSourceSignalSet, data)


class DiffuseSignalSetList(DiffuseSetList):
    def __init__(self, data=None):

        if data is not None and len(data) > 1:
            raise ValueError("The DiffuseSignalSetList can contain at most one dataset")
        else:
            super().__init__(DiffuseSignalSet, data)

    def __setitem__(self, i, v):
        self.check(v)

        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        if len(self._list) < 1:
            self._list[i] = v
        else:
            raise ValueError("The DiffuseSignalSetList can contain at most one dataset")

    def insert(self, i, v):
        self.check(v)
        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        if len(self._list) < 1:
            self._list.insert(i, v)
        else:
            raise ValueError("The DiffuseSignalSetList can contain at most one dataset")


class BackgroundSetList(DiffuseSetList):
    def __init__(self, data=None):
        super().__init__(BackgroundSet, data)
