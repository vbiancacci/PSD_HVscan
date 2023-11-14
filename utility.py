import numpy as np
import pygama.pargen.energy_optimisation as om #import get_peak_fwhm_with_dt_corr
from pygama.math.peak_fitting import extended_radford_pdf, radford_pdf, extended_gauss_step_pdf, gauss_step_pdf, goodness_of_fit

import logging
log = logging.getLogger(__name__)

def fom_FWHM_with_dt_corr_fit(energies, dt_eff, kwarg_dict, ctc_parameter, idxs=None, display=0):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm with its error,
    the corresponding alpha value and the number of events in the fitted peak, also the reduced chisquare of the
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    gof_func = kwarg_dict["gof_func"]
    Energies = energies #tb_in[parameter].nda
    Energies = Energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    min_alpha = 0
    max_alpha = 3.50e-06
    astep = 1.250e-07
    dt = dt_eff
    #if ctc_parameter == "QDrift":
    #    dt = tb_in["dt_eff"].nda
    #elif ctc_parameter == "dt":
    #    dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    #elif ctc_parameter == "rt":
    #    dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")

    if idxs is not None:
        Energies = Energies[idxs]
        dt = dt[idxs]

    if np.isnan(Energies).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }
    if np.isnan(dt).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    alphas = np.array(
        [
            0.000e00,
            1.250e-07,
            2.500e-07,
            3.750e-07,
            5.000e-07,
            6.250e-07,
            7.500e-07,
            8.750e-07,
            1.000e-06,
            1.125e-06,
            1.250e-06,
            1.375e-06,
            1.500e-06,
            1.625e-06,
            1.750e-06,
            1.875e-06,
            2.000e-06,
            2.125e-06,
            2.250e-06,
            2.375e-06,
            2.500e-06,
            2.625e-06,
            2.750e-06,
            2.875e-06,
            3.000e-06,
            3.125e-06,
            3.250e-06,
            3.375e-06,
            3.500e-06,
        ],
        dtype="float64",
    )
    fwhms = np.array([])
    final_alphas = np.array([])
    fwhm_errs = np.array([])
    guess = None
    best_fwhm = np.inf
    for alpha in alphas:
        (
            _,
            fwhm_o_max,
            _,
            fwhm_o_max_err,
            _,
            _,
            _,
            fit_pars,
        ) = om.get_peak_fwhm_with_dt_corr(
            Energies, alpha, dt, func, gof_func, peak, kev_width, guess=guess
        )
        if not np.isnan(fwhm_o_max):
            fwhms = np.append(fwhms, fwhm_o_max)
            final_alphas = np.append(final_alphas, alpha)
            fwhm_errs = np.append(fwhm_errs, fwhm_o_max_err)
            guess = fit_pars
            if fwhms[-1] < best_fwhm:
                best_fwhm = fwhms[-1]
                best_fit = fit_pars
        log.info(f"alpha: {alpha}, fwhm/max:{fwhm_o_max}+-{fwhm_o_max_err}")

    # Make sure fit isn't based on only a few points
    if len(fwhms) < 10:
        log.debug("less than 10 fits successful")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    ids = (fwhm_errs < 2 * np.nanpercentile(fwhm_errs, 50)) & (fwhm_errs > 0)
    # Fit alpha curve to get best alpha

    try:
        alphas = np.arange(
            final_alphas[ids][0], final_alphas[ids][-1], astep / 20, dtype="float64"
        )
        alpha_fit, cov = np.polyfit(
            final_alphas[ids], fwhms[ids], w=1 / fwhm_errs[ids], deg=4, cov=True
        )
        fit_vals = np.polynomial.polynomial.polyval(alphas, alpha_fit[::-1])
        alpha = alphas[np.nanargmin(fit_vals)]

        rng = np.random.default_rng(1)
        alpha_pars_b = rng.multivariate_normal(alpha_fit, cov, size=1000)
        fits = np.array(
            [
                np.polynomial.polynomial.polyval(alphas, pars[::-1])
                for pars in alpha_pars_b
            ]
        )
        min_alphas = np.array([alphas[np.nanargmin(fit)] for fit in fits])
        alpha_err = np.nanstd(min_alphas)
        if display > 0:
            plt.figure()
            yerr_boot = np.std(fits, axis=0)
            plt.errorbar(final_alphas, fwhms, yerr=fwhm_errs, linestyle=" ")
            plt.plot(alphas, fit_vals)
            plt.fill_between(
                alphas,
                fit_vals - yerr_boot,
                fit_vals + yerr_boot,
                facecolor="C1",
                alpha=0.5,
            )
            plt.show()

    except:
        log.debug("alpha fit failed")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if np.isnan(fit_vals).all():
        log.debug("alpha fit all nan")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    else:
        # Return fwhm of optimal alpha in kev with error
        (
            final_fwhm,
            _,
            final_err,
            _,
            csqr,
            n_sig,
            n_sig_err,
            _,
        ) = om.get_peak_fwhm_with_dt_corr(
            Energies,
            alpha,
            dt,
            func,
            gof_func,
            peak,
            kev_width,
            guess=best_fit,
            kev=True,
            display=display,
        )
        if np.isnan(final_fwhm) or np.isnan(final_err):
            (
                final_fwhm,
                _,
                final_err,
                _,
                csqr,
                n_sig,
                n_sig_err,
                _,
            ) = om.get_peak_fwhm_with_dt_corr(
                Energies,
                alpha,
                dt,
                func,
                gof_func,
                peak,
                kev_width,
                kev=True,
                display=display,
            )
        if np.isnan(final_fwhm) or np.isnan(final_err):
            log.debug(f"final fit failed, alpha was {alpha}")
        return {
            "fwhm": final_fwhm,
            "fwhm_err": final_err,
            "alpha": alpha,
            "alpha_err": alpha_err,
            "chisquare": csqr,
            "n_sig": n_sig,
            "n_sig_err": n_sig_err,
        }


def fom_FWHM(energies, dt_eff, kwarg_dict, ctc_parameter, alpha, idxs=None, display=0):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    cs_func = kwarg_dict["gof_func"]
    Energies = energies
    Energies = Energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]

    dt = dt_eff

    #if ctc_parameter == "QDrift":
    #    dt = tb_in["dt_eff"].nda
    #elif ctc_parameter == "dt":
    #    dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    #elif ctc_parameter == "rt":
    #    dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")
    
    if np.isnan(Energies).any() or np.isnan(dt).any():
        if np.isnan(Energies).any():
            log.debug(f"nan energy values for peak {peak}")
        else:
            log.debug(f"nan dt values for peak {peak}")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if idxs is not None:
        Energies = Energies[idxs]
        dt = dt[idxs]

    # Return fwhm of optimal alpha in kev with error
    try:
        (
            final_fwhm,
            _,
            final_err,
            _,
            csqr,
            n_sig,
            n_sig_err,
            _,
        ) = om.get_peak_fwhm_with_dt_corr(
            Energies,
            alpha,
            dt,
            func,
            cs_func,
            peak,
            kev_width,
            kev=True,
            display=display,
        )
    except:
        final_fwhm = np.nan
        final_err = np.nan
        csqr = np.nan
        n_sig = np.nan
        n_sig_err = np.nan
    return {
        "fwhm": final_fwhm,
        "fwhm_err": final_err,
        "alpha": alpha,
        "chisquare": csqr,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }



def new_fom(energies, dt_eff, kwarg_dict):
    peaks = kwarg_dict["peaks_keV"]
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"] #QDrift

    peak_dicts = kwarg_dict["peak_dicts"]

    out_dict = fom_FWHM_with_dt_corr_fit(
        energies, dt_eff, peak_dicts[-1], ctc_param, idxs=idx_list[-1], display=0
    )
    alpha = out_dict["alpha"]
    #log.info(alpha)
    fwhms = []
    fwhm_errs = []
    n_sig = []
    n_sig_err = []
    for i, peak in enumerate(peaks[:-1]):
        out_peak_dict = fom_FWHM(
            energies, dt_eff, peak_dicts[i], ctc_param, alpha, idxs=idx_list[i], display=0
        )
        # n_sig_minimum = peak_dicts[i]["n_sig_minimum"]
        # if peak_dict["n_sig"]<n_sig_minimum:
        #    out_peak_dict['fwhm'] = np.nan
        #   out_peak_dict['fwhm_err'] = np.nan
        fwhms.append(out_peak_dict["fwhm"])
        fwhm_errs.append(out_peak_dict["fwhm_err"])
        n_sig.append(out_peak_dict["n_sig"])
        n_sig_err.append(out_peak_dict["n_sig_err"])
    fwhms.append(out_dict["fwhm"])
    fwhm_errs.append(out_dict["fwhm_err"])
    n_sig.append(out_dict["n_sig"])
    n_sig_err.append(out_dict["n_sig_err"])
    #log.info(f"fwhms are {fwhms}keV +- {fwhm_errs}")
    #qbb, qbb_err, fit_pars = interpolate_energy(
    #    np.array(peaks), np.array(fwhms), np.array(fwhm_errs), 2039
    #)

    #log.info(f"Qbb fwhm is {qbb} keV +- {qbb_err}")

    return {
        #"y_val": qbb,
        #"y_err": qbb_err,
        #"qbb_fwhm": qbb,
        #"qbb_fwhm_err": qbb_err,
        #"alpha": alpha,
        "peaks": peaks.tolist(),
        "fwhms": fwhms,
        "fwhm_errs": fwhm_errs,
        #"n_events": n_sig,
        "n_sig_err": n_sig_err,
    }
