import astropy.units as u
from ctapipe.core import Component
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.data import Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom

from ctapipe.core.traits import List, Unicode, Int
from traitlets import Float


class PSSensitivityCalculator(Component):

    livetime = Unicode(
        default_value="50 h",
        help="Observation Time",
    ).tag(config=True)

    location_name = Unicode(
        default_value="cta_south",
        help="Name of the location of the observatory",
    ).tag(config=True)

    containment = Float(
        help="Percentile of events to keep",
        default_value=0.68,
    ).tag(config=True)

    offset = Float(
        help="FoV offset to calculate the sensitvity at in deg",
        default_value=0.0,
    ).tag(config=True)

    gamma_min = Int(
        help="Minimum number of gamma ray events for sensitivity",
        default_value=5,
    ).tag(config=True)

    n_sigma = Float(
        help="Number of sigma to use for the sensitvity curve",
        default_value=3.0,
    ).tag(config=True)

    bkg_syst_fraction = Float(
        help="Background relative systematic uncertainty",
        default_value=0.1,
    ).tag(config=True)

    n_reco_energy_bins = Int(
        help="Number of reco energy bins to evaluate sensitivity in ",
        default_value=20,
    ).tag(config=True)

    n_true_energy_bins = Int(
        help="Number of true energy bins",
        default_value=30,
    ).tag(config=True)

    true_energy_bounds = List(
        default_value=["0.05 TeV", "200 TeV"],
        minlen=2,
        maxlen=2,
        help="Minimum and maximum values of true energy",
    ).tag(config=True)

    reco_energy_bounds = List(
        default_value=["0.1 TeV", "100 TeV"],
        minlen=2,
        maxlen=2,
        help="Minimum and maximum values of reco energy",
    ).tag(config=True)

    alpha = Float(
        help="Ratio of ON to OFF region",
        default_value=0.2,
        allow_none=False,
    ).tag(config=True)

    def __init__(self) -> None:
        super().__init__()
        self.location = observatory_locations[self.location_name]
        self.pointing = SkyCoord(0 * u.deg, 0 * u.deg)

        self.energy_axis = MapAxis.from_energy_bounds(
            self.reco_energy_bounds[0],
            self.reco_energy_bounds[1],
            nbin=self.n_reco_energy_bins,
        )

        energy_axis_true = MapAxis.from_energy_bounds(
            self.true_energy_bounds[0],
            self.true_energy_bounds[1],
            nbin=self.n_true_energy_bins,
            name="energy_true",
        )

        self.geom = RegionGeom.create(
            "icrs;circle(0, {}, 0.5)".format(round(self.offset, 2)),
            axes=[self.energy_axis],
        )

        self.empty_dataset = SpectrumDataset.create(
            geom=self.geom, energy_axis_true=energy_axis_true
        )

        self.spectrum_maker = SpectrumDatasetMaker(
            selection=["exposure", "edisp", "background"]
        )

        self.sensitivity_estimator = SensitivityEstimator(
            gamma_min=5, n_sigma=3, bkg_syst_fraction=0.10
        )

    def __call__(self, irf_handler):

        assert irf_handler.aeff is not None
        assert irf_handler.psf is not None
        assert irf_handler.edisp is not None
        assert irf_handler.bkg_model is not None

        irf_dict = {
            "aeff": irf_handler.aeff,
            "psf": irf_handler.psf,
            "edisp": irf_handler.edisp,
            "bkg": irf_handler.bkg_model,
        }

        obs = Observation.create(
            pointing=self.pointing,
            irfs=irf_dict,
            livetime=self.livetime,
            location=self.location,
        )
        spectrum_maker = SpectrumDatasetMaker(
            selection=["exposure", "edisp", "background"]
        )
        dataset = spectrum_maker.run(self.empty_dataset, obs)

        dataset.exposure *= self.containment

        # correct background estimation
        on_radii = obs.psf.containment_radius(
            energy_true=self.energy_axis.center,
            offset=self.offset * u.deg,
            fraction=self.containment,
        )
        factor = (1 - np.cos(on_radii)) / (1 - np.cos(self.geom.region.radius))
        dataset.background *= factor.value.reshape((-1, 1, 1))

        dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
            dataset=dataset, acceptance=1, acceptance_off=1 / self.alpha
        )
        sensitivity_estimator = SensitivityEstimator(
            gamma_min=self.gamma_min,
            n_sigma=self.n_sigma,
            bkg_syst_fraction=self.bkg_syst_fraction,
        )
        sensitivity_table = sensitivity_estimator.run(dataset_on_off)

        return sensitivity_table

    @staticmethod
    def plot_sensitvity_curve(ax, sensitivity_table):

        t = sensitivity_table

        is_s = t["criterion"] == "significance"

        ax.plot(
            t["energy"][is_s],
            t["e2dnde"][is_s],
            "s-",
            color="red",
            label="significance",
        )

        is_g = t["criterion"] == "gamma"
        ax.plot(t["energy"][is_g], t["e2dnde"][is_g], "*-", color="blue", label="gamma")
        is_bkg_syst = t["criterion"] == "bkg"
        ax.plot(
            t["energy"][is_bkg_syst],
            t["e2dnde"][is_bkg_syst],
            "v-",
            color="green",
            label="bkg syst",
        )

        ax.loglog()
        ax.set_xlabel(f"Energy [{t['energy'].unit}]")
        ax.set_ylabel(f"Sensitivity [{t['e2dnde'].unit}]")
        ax.legend()
