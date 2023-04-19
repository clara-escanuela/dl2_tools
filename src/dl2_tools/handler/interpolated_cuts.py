from gammapy.maps import MapAxis
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy.table import QTable
import operator
import dill as pickle


def geq(a, b):
    return a >= b


def leq(a, b):
    return a <= b


class InterpolatedCut:
    def __init__(
        self,
        offset_axis,
        cut_column,
        cut_values,
        bin_axes=None,
        op=operator.ge,
        cut_op=lambda x: x,
        log_cut_value=False,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
    ):

        self.cut_column = cut_column

        bin_axes_points = (offset_axis.center,)

        if bin_axes is not None:
            self.bin_columns = [bin_axis.name for bin_axis in bin_axes]
            assert len(bin_axes) + 1 == len(np.shape(cut_values))
            for bin_axis in bin_axes:
                if isinstance(bin_axis.center, u.Quantity):
                    if bin_axis.center.unit.is_equivalent(u.TeV):
                        bin_axes_points += (np.log10(bin_axis.center.to_value(u.TeV)),)
                    else:
                        bin_axes_points += (bin_axis.center,)
                else:
                    if not u.Quantity(bin_axis.center).unit.is_equivalent(u.TeV):
                        bin_axes_points += (bin_axis.center,)
                    else:
                        bin_axes_points += (np.log10(bin_axis.center.to_value(u.TeV)),)
        else:
            self.bin_columns = None
            assert len(np.shape(cut_values)) == 1

        if not log_cut_value:
            cut_value_points = cut_values
        else:
            cut_value_points = np.log10(cut_values)

        self.log_cut = log_cut_value

        self.cut_spline = RegularGridInterpolator(
            bin_axes_points,
            cut_value_points,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        self.op = op
        self.cut_op = cut_op

    @classmethod
    def from_cut_tables(
        cls,
        cut_tables,
        offset_axis,
        bin_axis_name,
        cut_column,
        op=geq,
        cut_op=lambda x: x,
        log_cut_value=False,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
    ):

        assert len(offset_axis.center) == len(cut_tables)

        cut_values = np.empty((len(offset_axis.center), len(cut_tables[0])))

        bin_axes_edges = u.Quantity(
            np.concatenate(
                (cut_tables[0]["low"], np.atleast_1d(cut_tables[0]["high"][-1]))
            )
        )

        bin_axes = MapAxis(
            nodes=bin_axes_edges, interp="log", name=bin_axis_name, node_type="edges"
        )

        for i, cut_table in enumerate(cut_tables):
            assert len(cut_table) == len(cut_tables[0])
            assert (cut_table["low"] == cut_tables[0]["low"]).all()
            assert (cut_table["high"] == cut_tables[0]["high"]).all()
            cut_values[i] = cut_table["cut"]

        return cls(
            offset_axis,
            cut_column,
            cut_values,
            [bin_axes],
            op,
            cut_op,
            log_cut_value,
            method,
            bounds_error,
            fill_value,
        )

    def to_cut_table(self, offset, bin_axis):
        assert len(self.bin_columns) == 1

        cut_table = QTable()
        cut_table["low"] = bin_axis.edges[:-1]
        cut_table["high"] = bin_axis.edges[1:]

        if bin_axis.center.unit.is_equivalent(u.TeV):
            interp_values = np.log10(bin_axis.center.to_value(u.TeV))
        else:
            interp_values = bin_axis.center

        interp_array = np.array([np.repeat(offset, len(interp_values)), interp_values])

        if not self.log_cut:
            cut_values = self.cut_spline(interp_array.T)
        else:
            cut_values = 10 ** self.cut_spline(interp_array.T)

        cut_table["cut"] = cut_values

        return cut_table

    def writeto(self, filepath):

        name = filepath
        file = open(name, "wb")
        pickle.dump(self, file)
