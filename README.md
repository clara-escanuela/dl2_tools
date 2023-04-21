This package conains and wraps tools to produce dl3 data from dl2 data as produced by ctapipe.

Its main functionalities include:

- Calculate and apply g/h cuts
- Calculate cuts on essentially all other variables
- Calculate and plot irfs
- Make ROC plots

This can be done out of the box both with point-like and diffuse gamma ray simulations.

Currently there are two command line tools similar to the ctapipe tools. 

The first one calculates IRFs and optionally the ROC plot from dl2 files. It can be invoked as follows:

irf-maker --signal-type PointSource --signal-input /your/ctapipe/dl2/signal/file.h5 --background-input /your/ctapipe/dl2/background/file.h5 -o /directory/in/which/you/store/irfs/ --geometry-reco YourGeometryReconstructor --energy-reco YourEnergyReconstructor --gh-score YourGHSeparator --cuts-files /your/saved/interpolated_cuts_file.cut --save-plots --overwrite --make-roc --IRFMakerTool.signal_weight_name CRAB_HEGRA --IRFMakerTool.background_weight_name IRFDOC_PROTON_SPECTRUM

For further commandline options see dl_tools/tools/irf_tool.py

The second one allows you to calculate a binned percentile cut on a quantity/column as a function of a different quantity/column.

Potential usage for calculating a cut that keeps only the events in the upper 10% quantile of triggered telescopes. 

percentile-cut-maker --input-type PointSource --input-files /your/ctapipe/dl2/signal/file.h5 -o -o /directory/in/which/you/store/cuts/ --geometry-reco ImPACTReconstructor --energy-reco ImPACTReconstructor --gh-score RFCl_100_est_dpth_20_int_wgt --PercentileCutCalculatorTool.offset_axis_name reco_fov_source_offset --PercentileCutCalculator.cut_variable "tels_with_trigger" --PercentileCutCalculator.cut_op_name "lambda x: np.sum(x,axis=1)" --PercentileCutCalculator.percentile 10

dl2_tools can also be used in jupyter notebboks and such, see the examples.