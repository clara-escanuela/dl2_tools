from ctapipe.core import Tool, traits

from traitlets import Float

from ctapipe.core.traits import List, Unicode, classes_with_traits, Bool, flag

from ..handler.sim_datasets import PointSourceSignalSet,BackgroundSet

from ..handler.binning import IRFBinning

from ..maker.cut_optimizer import (
    PercentileCutCalculator,
    CutCalculator,
    RecoEnergyPointSourceGHCutOptimizer,
)
import astropy.units as u
import dill as pickle

from ..handler.observation_handler import PointSourceObservationHandler

from pyirf.spectral import (
    PowerLaw,
    LogParabola,
    PowerLawWithExponentialGaussian,
    POINT_SOURCE_FLUX_UNIT,
    DIFFUSE_FLUX_UNIT,
    CRAB_HEGRA,
    CRAB_MAGIC_JHEAP2015,
    PDG_ALL_PARTICLE,
    IRFDOC_PROTON_SPECTRUM,
    DAMPE_P_He_SPECTRUM,
)


class PSSensitivityCutOptimizerTool(Tool):

    """
    Calculate a gh cut optimized for sensitivity to a point source at a given fov offset
    """

    name = "ps-sensitivity-optimized-cut-maker"

    cuts_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Files of pickled InterpolatedCuts",
    ).tag(config=True)

    signal_input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl2 files",
    ).tag(config=True)

    background_input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl2 files",
    ).tag(config=True)

    output_file_radius_cut = traits.Path(
        help="Output file for radius cut",
        file_ok=True,
    ).tag(config=True)

    output_file_gh_cut = traits.Path(
        help="Output file for radius cut",
        file_ok=True,
    ).tag(config=True)

    energy_reco = Unicode(
        default_value="ImPACTReconstructor",
        help="Name of energy reco",
    ).tag(config=True)

    geometry_reco = Unicode(
        default_value="HillasReconstructor",
        help="Name of geometry reco",
    ).tag(config=True)

    gh_score = Unicode(
        help="Name of gh score",
    ).tag(config=True)

    signal_weight_name = Unicode(
        help="Name of the signal weights. Must be in pyirf spectral",
        default="CRAB_HEGRA",
        allow_none=False,
    ).tag(config=True)

    background_weight_name = Unicode(
        help="Name of the signal weights. Must be in pyirf spectral",
        default="IRFDOC_PROTON_SPECTRUM",
        allow_none=False,
    ).tag(config=True)

    signal_powerlaw_params = List(
        Float,
        default_value=[1e-11, 2],
        minlen=2,
        maxlen=2,
        help="Parameters to weight signal to a powerlaw spectrum",
    ).tag(config=True)

    background_powerlaw_params = List(
        Float,
        default_value=[1e-5, 2.7],
        minlen=2,
        maxlen=2,
        help="Parameters to weight baclground to a powerlaw spectrum",
    ).tag(config=True)

    signal_logparabola_params = List(
        Float,
        default_value=[1e-11, 2, -0.2],
        minlen=3,
        maxlen=3,
        help="Parameters to weight signal to a logparabola spectrum",
    ).tag(config=True)

    background_logparabola_params = List(
        Float,
        default_value=[1e-5, 2.7, 0.0],
        minlen=3,
        maxlen=3,
        help="Parameters to weight background to a logparabola spectrum",
    ).tag(config=True)

    signal_plwithexpgauss_params = List(
        Float,
        default_value=[1e-11, 2, 0, 1, 1],
        minlen=5,
        maxlen=5,
        help="Parameters to weight signal to a plwithexpgauss spectrum",
    ).tag(config=True)

    background_plwithexpgauss_params = List(
        Float,
        default_value=[1e-5, 2.7, 0.0, 1, 1],
        minlen=5,
        maxlen=5,
        help="Parameters to weight background to a plwithexpgauss spectrum",
    ).tag(config=True)

    overwrite = Bool(
        help="Overwrite saved IRFs?",
        default_value=False,
    ).tag(config=True)

    aliases = {
        "signal-input": "PSSensitivityCutOptimizerTool.signal_input_files",
        "background-input": "PSSensitivityCutOptimizerTool.background_input_files",
        "gh-output": "PSSensitivityCutOptimizerTool.output_file_gh_cut",
        "radius-output": "PSSensitivityCutOptimizerTool.output_file_radius_cut",
        "geometry-reco": "PSSensitivityCutOptimizerTool.geometry_reco",
        "energy-reco": "PSSensitivityCutOptimizerTool.energy_reco",
        "gh-score": "PSSensitivityCutOptimizerTool.gh_score",
    }

    flags = {
        **flag(
            "overwrite",
            "PSSensitivityCutOptimizerTool.overwrite",
            "Overwrite saved IRFs",
            "Do not overwrite saved IRFs",
        ),
    }

    classes = (
        [
            RecoEnergyPointSourceGHCutOptimizer,
        ]
        + classes_with_traits(RecoEnergyPointSourceGHCutOptimizer)
    )

    def setup(self):

        self.observation = PointSourceObservationHandler()
        assert len(self.background_input_files) > 0
        for inp_file in self.signal_input_files:
            self.observation.add_signal(
                PointSourceSignalSet.from_path(
                    inp_file, self.energy_reco, self.geometry_reco, self.gh_score
                )
            )


        
        match self.signal_weight_name:
            case "PowerLaw":
                if self.signal_type == "PointSource":
                    signal_spectrum=PowerLaw(normalization=self.signal_powerlaw_params[0]*POINT_SOURCE_FLUX_UNIT,index=self.signal_powerlaw_params[1])
                else:
                    signal_spectrum=PowerLaw(normalization=self.signal_powerlaw_params[0]*DIFFUSE_FLUX_UNIT,index=self.signal_powerlaw_params[1])
            case "LogParabola":
                if self.signal_type == "PointSource":
                    signal_spectrum=LogParabola(normalization=self.signal_logparabola_params[0]*POINT_SOURCE_FLUX_UNIT,a=self.signal_logparabola_params[1],b=self.signal_logparabola_params[2])
                else:
                    signal_spectrum=LogParabola(normalization=self.signal_logparabola_params[0]*DIFFUSE_FLUX_UNIT,a=self.signal_logparabola_params[1],b=self.signal_logparabola_params[2])
            case "PowerLawWithExponentialGaussian":
                if self.signal_type == "PointSource":
                    signal_spectrum=PowerLawWithExponentialGaussian(normalization=self.signal_plwithexpgauss_params[0]*POINT_SOURCE_FLUX_UNIT,index=self.signal_plwithexpgauss_params[1],f=self.signal_plwithexpgauss_params[2],mu=self.signal_plwithexpgauss_params[3]*u.TeV,sigma=self.signal_plwithexpgauss_params[4]*u.TeV)
                else:
                    signal_spectrum=PowerLawWithExponentialGaussian(normalization=self.signal_plwithexpgauss_params[0]*DIFFUSE_FLUX_UNIT,index=self.signal_plwithexpgauss_params[1],f=self.signal_plwithexpgauss_params[2],mu=self.signal_plwithexpgauss_params[3]*u.TeV,sigma=self.signal_plwithexpgauss_params[4]*u.TeV)
            case "CRAB_HEGRA":
                signal_spectrum=CRAB_HEGRA
            case "CRAB_MAGIC_JHEAP_2015":
                signal_spectrum=CRAB_MAGIC_JHEAP2015
            case _:
                raise ValueError("signal_weight_name is not among the allowed names")

        self.observation.signal.reweight_to(signal_spectrum)

        assert len(self.background_input_files) > 0
        for inp_file in self.background_input_files:
            self.observation.add_background(
                BackgroundSet.from_path(
                    inp_file, self.energy_reco, self.geometry_reco, self.gh_score
                )
            )

        
        match self.background_weight_name:
            case "PowerLaw":
                    background_spectrum=PowerLaw(normalization=self.background_powerlaw_params[0]*DIFFUSE_FLUX_UNIT,index=self.background_powerlaw_params[1])
            case "LogParabola":
                    background_spectrum=LogParabola(normalization=self.background_logparabola_params[0]*DIFFUSE_FLUX_UNIT,a=self.background_logparabola_params[1],b=self.background_logparabola_params[2])
            case "PowerLawWithExponentialGaussian":
                    background_spectrum=PowerLawWithExponentialGaussian(normalization=self.background_plwithexpgauss_params[0]*DIFFUSE_FLUX_UNIT,index=self.background_plwithexpgauss_params[1],f=self.background_plwithexpgauss_params[2],mu=self.background_plwithexpgauss_params[3]*u.TeV,sigma=self.background_plwithexpgauss_params[4]*u.TeV)
            case "PDG_ALL_PARTICLE":
                background_spectrum=PDG_ALL_PARTICLE
            case "IRFDOC_PROTON_SPECTRUM":
                background_spectrum=IRFDOC_PROTON_SPECTRUM
            case "DAMPE_P_He_Spectrum":
                background_spectrum=DAMPE_P_He_SPECTRUM
            case _:
                raise ValueError("background_weight_name is not among the allowed names")

        self.observation.background.reweight_to(background_spectrum)

        cuts = []

        for cut_file in self.cuts_files:
            cuts.append(pickle.load(open(cut_file, "rb")))

        if len(cuts) > 0:
            self.observation.set_signal_cuts(cuts)
            self.observation.set_background_cuts(cuts)

        self.cut_calculator = RecoEnergyPointSourceGHCutOptimizer(parent=self)

    def start(self):

        self.gh_cut, self.radius_cut = self.cut_calculator(self.observation)

    def finish(self):

        self.radius_cut.writeto(self.output_file_radius_cut)
        self.gh_cut.writeto(self.output_file_gh_cut)

def main():
    """run the tool"""
    tool = PSSensitivityCutOptimizerTool()
    tool.run()


if __name__ == "__main__":
    main()

