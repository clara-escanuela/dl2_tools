# Contains the irfs for a given set of metaparameters (zenith,EnergyReconstructor,...) and
# g/h cut

import os
import astropy.units as u


class IRFHandler:
    """
    Class that handles all irfs and has methods to save them to file in gammapy formats
    """

    def __init__(
        self, aeff=None, edisp=None, psf=None, bkg_model=None, binning=None
    ) -> None:
        """
        Initialize the IRFHandler. Passing IRFs is optional. They can be set later.

        Parameters
        ----------
        aeff : gammapy.irf.EffectiveArea2DTable, optional
            Effective area in gammapy format, by default None
        edisp : gammapy.irf.EnergyDispersion2D, optional
            Energy dispersion in gammapy format, by default None
        psf : gammapy.irf.PSF3D, optional
            Point Spread Function in gammapy format, by default None
        bkg_model : gammapy.IRF.Background2D, optional
            Radially symmetric background in gammapy format, by default None
        binning : binning.IRFBinning, optional
            An IRFBinning object that contains all relevant axes, by default None
        """
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg_model = bkg_model

        # Metadata:
        # zenith angle

        # Binning
        self.binning = binning

    def save_bkg_model_fits(self, savedir, suffix="", overwrite=False):
        assert (
            self.bkg_model is not None
        ), "You first need to set a background model before saving it"

        if not os.path.exists(os.path.join(savedir, "bkg_model/")):
            os.mkdir(os.path.join(savedir, "bkg_model/"))

        self.bkg_model.write(
            os.path.join(
                savedir,
                "bkg_model/bkg_reco_{}TeV_{}TeV_{}bins_fov_off_{}deg_{}deg_{}bins_{}.fits".format(
                    self.binning.energy_reco.edges[0].to_value(u.TeV).round(2),
                    self.binning.energy_reco.edges[-1].to_value(u.TeV).round(2),
                    len(self.binning.energy_reco.center),
                    self.binning.background_offset.edges[0].to_value(u.deg).round(2),
                    self.binning.background_offset.edges[-1].to_value(u.deg).round(2),
                    len(self.binning.background_offset.center),
                    suffix,
                ),
            ),
            overwrite=overwrite,
        )

    def save_Aeff_fits(self, savedir, suffix="", overwrite=False):

        assert (
            self.aeff is not None
        ), "You first need to set an effective area before saving it"

        if not os.path.exists(os.path.join(savedir, "aeffs/")):
            os.mkdir(os.path.join(savedir, "aeffs/"))

        self.aeff.write(
            os.path.join(
                savedir,
                "aeffs/Aeff_true_{}TeV_{}TeV_{}bins_fov_off_{}deg_{}deg_{}bins_{}.fits".format(
                    self.binning.energy_true.edges[0].to_value(u.TeV).round(2),
                    self.binning.energy_true.edges[-1].to_value(u.TeV).round(2),
                    len(self.binning.energy_true.center),
                    self.binning.signal_offset.edges[0].to_value(u.deg).round(2),
                    self.binning.signal_offset.edges[-1].to_value(u.deg).round(2),
                    len(self.binning.signal_offset.center),
                    suffix,
                ),
            ),
            overwrite=overwrite,
        )

    def save_Edisp_fits(self, savedir, suffix="", overwrite=False):

        assert (
            self.edisp is not None
        ), "You first need to set an energy dispersion before saving it"

        if not os.path.exists(os.path.join(savedir, "edisps/")):
            os.mkdir(os.path.join(savedir, "edisps/"))

        self.edisp.write(
            os.path.join(
                savedir,
                "edisps/Edisp_true_{}TeV_{}TeV_{}bins_reco_{}_{}_{}bins_fov_off_{}deg_{}deg_{}bins_{}.fits".format(
                    self.binning.energy_true.edges[0].to_value(u.TeV).round(2),
                    self.binning.energy_true.edges[-1].to_value(u.TeV).round(2),
                    len(self.binning.energy_true.center),
                    self.binning.migration.edges[0].round(2),
                    self.binning.migration.edges[-1].round(2),
                    len(self.binning.migration.center),
                    self.binning.signal_offset.edges[0].to_value(u.deg).round(2),
                    self.binning.signal_offset.edges[-1].to_value(u.deg).round(2),
                    len(self.binning.signal_offset.center),
                    suffix,
                ),
            ),
            overwrite=overwrite,
        )

    def save_PSF_fits(self, savedir, suffix="", overwrite=False):

        assert (
            self.psf is not None
        ), "You first need to set an point spread function before saving it"

        if not os.path.exists(os.path.join(savedir, "psfs/")):
            os.mkdir(os.path.join(savedir, "psfs/"))

        self.psf.write(
            os.path.join(
                savedir,
                "psfs/PSF_true_{}TeV_{}TeV_{}bins_psf_off_{}deg_{}deg_{}bins_fov_off_{}deg_{}deg_{}bins_{}.fits".format(
                    self.binning.energy_true.edges[0].to_value(u.TeV).round(2),
                    self.binning.energy_true.edges[-1].to_value(u.TeV).round(2),
                    len(self.binning.energy_true.center),
                    self.binning.psf_offset.edges[0].to_value(u.deg).round(2),
                    self.binning.psf_offset.edges[-1].to_value(u.deg).round(2),
                    len(self.binning.psf_offset.center),
                    self.binning.signal_offset.edges[0].to_value(u.deg).round(2),
                    self.binning.signal_offset.edges[-1].to_value(u.deg).round(2),
                    len(self.binning.signal_offset.center),
                    suffix,
                ),
            ),
            overwrite=overwrite,
        )
