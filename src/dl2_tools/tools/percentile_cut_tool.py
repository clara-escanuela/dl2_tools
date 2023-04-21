from ctapipe.core import Tool, traits

from ctapipe.core.traits import List, Unicode, classes_with_traits, Bool, flag

from ..handler.sim_datasets import PointSourceSignalSet, DiffuseSignalSet

from ..handler.binning import IRFBinning

from ..maker.cut_optimizer import PercentileCutCalculator, CutCalculator

from ..handler.data_lists import DiffuseSignalSetList, PointSourceSignalSetList


class PercentileCutMakerTool(Tool):
    """
    Calculate and save a binned cut with a given efficiency
    """

    name = "percentile-cut-maker"

    input_type = Unicode(
        default_value="PointSource",
        help="Type of input file",
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl2 files",
    ).tag(config=True)

    output_file = traits.Path(
        help="Output directory",
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

    overwrite = Bool(
        help="Overwrite saved IRFs?",
        default_value=False,
    ).tag(config=True)

    offset_axis_name = Unicode(
        help="Name of the offset axis to evaluate the cut on",
        default_value="true_source_fov_offset",
    ).tag(config=True)

    aliases = {
        "input-type": "PercentileCutMakerTool.input_type",
        "input-files": "PercentileCutMakerTool.input_files",
        ("o", "output"): "PercentileCutMakerTool.output_file",
        "geometry-reco": "PercentileCutMakerTool.geometry_reco",
        "energy-reco": "PercentileCutMakerTool.energy_reco",
        "gh-score": "PercentileCutMakerTool.gh_score",
    }

    flags = {
        **flag(
            "overwrite",
            "PercentileCutMakerTool.overwrite",
            "Overwrite saved IRFs",
            "Do not overwrite saved IRFs",
        ),
    }

    classes = (
        [CutCalculator, PercentileCutCalculator, IRFBinning]
        + classes_with_traits(CutCalculator)
        + classes_with_traits(PercentileCutCalculator)
        + classes_with_traits(IRFBinning)
    )

    def setup(self):

        if self.input_type == "PointSource":

            self.datalist = PointSourceSignalSetList()

            for inp_file in self.input_files:
                self.datalist.append(
                    PointSourceSignalSet.from_path(
                        inp_file, self.energy_reco, self.geometry_reco, self.gh_score
                    )
                )

        elif self.input_type == "Diffuse":
            assert len(self.input_files) == 1

            inp_set = DiffuseSignalSet.from_path(
                self.input_files[0],
                self.energy_reco,
                self.geometry_reco,
                self.gh_score,
            )

            self.datalist = DiffuseSignalSetList(data=inp_set)

        else:
            raise ValueError("input_type must be either PointSource or Diffuse")

        self.cut_calculator = PercentileCutCalculator(parent=self)

    def start(self):

        self.cut = self.cut_calculator(self.datalist, self.offset_axis_name)

    def finish(self):

        self.cut.writeto(self.output_file)


def main():
    """run the tool"""
    tool = PercentileCutMakerTool()
    tool.run()


if __name__ == "__main__":
    main()
