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
        Initialize a DataList

        Parameters
        ----------
        type : descriptor of sim_datasets.DataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of DataSet
        data : (List of) sim_datasets.DataSet , optional
            (List of) subclass of DataSet to be conatined in DataList, by default None.
            Default creates empty list
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
        """Check if an object is of the type stored in the list"""
        if not isinstance(v, self.type):
            raise TypeError(v)

    def __setitem__(self, i, v):
        """Set item at position i in the list to v"""
        self.check(v)

        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        self._list[i] = v

    def insert(self, i, v):
        """Insert item v at position i in the list"""
        self.check(v)
        if self.pointing_direction is None:
            self.pointing_direction = v.pointing_direction
        else:
            assert self.pointing_direction == v.pointing_direction

        self._list.insert(i, v)

    def reweight_to(self, new_spectrum):
        """Reweight each entry in list to new_spectrum"""
        for entry in self._list:
            entry.reweight_to(new_spectrum)

    def set_obs_time(self, obs_time):
        """Set observation time for each entry in list to obs_time"""
        for entry in self._list:
            entry.set_obs_time(obs_time)

    def set_cuts(self, cuts, offset_column):
        """
        Set cuts for each entry in list

        Parameters
        ----------
        cuts : List of handler.interpolated_cuts.InterpolatedCut
        offset_column : str
            Name of the column in field-of-view offset to evaluate this cut on.
            Typically either ```true_source_fov_offset``` or ```reco_source_fov_offset```.
        """
        for entry in self._list:
            entry.set_cuts(cuts, offset_column)

    def add_cut(self, cut, offset_column):
        """
        Add a cut to each entry in list

        Parameters
        ----------
        cut : handler.interpolated_cuts.InterpolatedCut
        offset_column : str
            Name of the column in field-of-view offset to evaluate this cut on.
            Typically either ```true_source_fov_offset``` or ```reco_source_fov_offset```.
        """
        for entry in self._list:
            entry.add_cut(cut, offset_column)


class PointSourceSetList(DataList):
    """
    A DataList that contains point-source datasets.
    The two important features of this are:

    - There can only be one dataset at a fixed field-of-view offset
    - The datasets are ordered by their field-of-view offset
    """

    def __init__(self, type, data=None):
        """
        Initialize a PointSourceSetList

        Parameters
        ----------
        type : descriptor of sim_datasets.PointSourceDataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of PointSourceDataSet
        data : (List of) sim_datasets.PointSourceDataSet , optional
            (List of) subclass of DataSet to be conatined in PointSourceDataList, by default None.
            Default creates empty list
        """
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
        """Overwrites method of parent class, asserting two main features of PointSourceSetList"""
        assert v.offset not in self.get_offsets()
        super().__setitem__(i, v)
        self._list.sort(key=lambda e: e.offset)

    def insert(self, i, v):
        """Overwrites method of parent class, asserting two main features of PointSourceSetList"""
        assert v.offset not in self.get_offsets()
        super().insert(i, v)
        self._list.sort(key=lambda e: e.offset)

    def get_offsets(self):
        """
        Get the offsets of the contained datasets
        Returns
        -------
        u.Quantity
            Offsets as u.Quantity
        """
        return u.Quantity([entry.offset for entry in self._list])

    def get_source_positions(self):
        """
        Get the source positions of the point sources for each set
        Returns
        -------
        list of astropy.coordinates.SkyCoord
        """
        return [entry.source_position for entry in self._list]


class DiffuseSetList(DataList):
    """
    A DataList that contains diffuse datasets.
    """

    def __init__(self, type, data=None):
        """
        Initialize a DiffuseSetList

        Parameters
        ----------
        type : descriptor of sim_datasets.DiffuseDataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of DiffuseDataSet
        data : (List of) sim_datasets.DiffuseDataSet , optional
            (List of) subclass of DataSet to be conatined in DiffuseDataList, by default None.
            Default creates empty list
        """
        assert issubclass(type, DiffuseDataset)
        super().__init__(type, data)

    def get_radii(self):
        """
        Get the viewcone radii of the contained datasets
        Returns
        -------
        u.Quantity
            Radii as u.Quantity
        """
        return u.Quantity([entry.radius for entry in self._list])


class PointSourceSignalSetList(PointSourceSetList):
    """
    A DataList that contains point-like signal datasets.
    Essentially just a copy of PointSourceSetList, but the type contained
    in the list is enforced stronger.
    """

    def __init__(self, data=None):
        """
        Initialize a PointSourceSignalSetList

        Parameters
        ----------
        type : descriptor of sim_datasets.PointSourceSignalDataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of PointSourceSignalDataSet
        data : (List of) sim_datasets.PointSourceSignalDataSet , optional
            (List of) subclass of DataSet to be conatined in PointSourceSignalDataList, by default None.
            Default creates empty list
        """
        super().__init__(PointSourceSignalSet, data)


class DiffuseSignalSetList(DiffuseSetList):
    """
    A DataList that contains diffuse signal datasets.
    There are two changes compared to PointSourceSetList:

    -The type contained in the list is enforced stronger
    -The length of the list is at most one

    The latter ensures the correct weighting of the events.
    """

    def __init__(self, data=None):
        """
        Initialize a DiffuseSignalSetList

        Parameters
        ----------
        type : descriptor of sim_datasets.DiffuseSignalDataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of DiffuseSignalDataSet
        data : (List of) sim_datasets.DiffuseSignalDataSet , optional
            (List of) subclass of DataSet to be conatined in DiffuseSignalDataList, by default None.
            Default creates empty list
        """
        if data is not None and len(data) > 1:
            raise ValueError("The DiffuseSignalSetList can contain at most one dataset")
        else:
            super().__init__(DiffuseSignalSet, data)

    def __setitem__(self, i, v):
        """Overwrites method of parent class, asserting length to be at most one"""
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
        """Overwrites method of parent class, asserting length to be at most one"""
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
    """
    A DataList that contains diffuse background datasets. Essentially just a copy of DiffuseSetList, but the type contained
    in the list is enforced stronger.
    """

    def __init__(self, data=None):
        """
        Initialize a BackgroundSetList

        Parameters
        ----------
        type : descriptor of sim_datasets.BackgroundDataSet subclass
            Type contained in the list.
            Descriptor of a specific subclass of BackgroundDataSet
        data : (List of) sim_datasets.BackgroundDataSet , optional
            (List of) subclass of DataSet to be conatined in BackgroundDataList, by default None.
            Default creates empty list
        """
        super().__init__(BackgroundSet, data)
