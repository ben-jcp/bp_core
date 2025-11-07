import numpy as np
from scipy.integrate import trapezoid


_h = 6.62607015e-34  # J s
_c = 299792458. # m s-1
_k_B = 1.380649e-23  # J K-1


def func_wn(wnums, temps, units_out='mW'):
    """
    wavenumber units: cm-1
    temperature units: K
        if single T value passed to function, will calculate Planck
        function at this temperature throughout
    radiance units out: mW/(m2 sr cm-1) or LBLRTM W/(cm2 sr cm-1)
    """
    c1 = 1.e11 * 2 * _h * _c**2  # mW / (m2 sr cm-4)
    c2 = 1.e2 * _h * _c / _k_B  # K cm

    if units_out == 'mW':
        pf = 1.
    elif units_out == 'W':
        pf = 1.e-7  # to convert flux units from mW/m^2 to W/cm^2

    radiances = pf * np.divide(c1 * wnums**3,
                               np.exp(np.divide(c2 * wnums,
                                                temps)) - 1.)
    
    return radiances


def func_wl(wlens, temps):
    """
    wavelength units: um
    temperature units: K
        if single T value passed to function, will calculate Planck
        function at this temperature throughout
    radiance units out: W/(m2 sr um)
    """
    c1 = 1.e24 * 2 * _h * _c**2  # W / (m2 sr um-4)
    c2 = 1.e6 * _h * _c / _k_B  # K um

    radiances = np.divide(c1,
                          wlens**5 * (np.exp(np.divide(c2,
                                                       wlens * temps)) - 1.))
    
    return radiances


def inverse_wn(spectrum, wnums, units_in='mW'):
    """
    wavenumber units: cm-1
    radiance units: mW/(m2 sr cm-1) or LBLRTM W/(cm2 sr cm-1)
    """
    c1 = 1.e11 * 2 * _h * _c**2  # mW / (m2 sr cm-4)
    c2 = 1.e2 * _h * _c / _k_B  # K cm

    if units_in == 'mW':
        pf = 1.
    elif units_in == 'W':
        pf = 1.e7  # to convert flux units from W/cm^2 to mW/m^2

    radiances = pf * np.asarray(spectrum)
    brightness_temp = np.divide(c2 * wnums,
                                np.log(1 + np.divide(c1 * wnums**3,
                                                     radiances)))
    
    return brightness_temp


def inverse_wl(spectrum, wlens):
    """
    wavelength units: um
    radiance units: W/(m2 sr um)
    """
    c1 = 1.e24 * 2 * _h * _c**2  # W / (m2 sr um-4)
    c2 = 1.e6 * _h * _c / _k_B  # K um
    
    radiances = np.asarray(spectrum)
    brightness_temp = np.divide(np.divide(c2, wlens),
                                np.log(1 + np.divide(c1,
                                                     wlens**5 * radiances)))
    
    return brightness_temp


def nedt_wn(nedrs, radiances, wnums):
    """
    wavenumber units: cm-1
    radiance units: mW/(m2 sr cm-1)
    """
    c1 = 1.e11 * 2 * _h * _c**2  # mW / (m2 sr cm-4)
    c2 = 1.e2 * _h * _c / _k_B  # K cm
    
    c1_factor = np.divide(c1 * wnums**3, radiances)

    nedts = np.divide(c2 * wnums,
                      np.log(1 + c1_factor)**2) * np.divide(c1_factor,
                                                            radiances * (1 + c1_factor)) * nedrs
    
    return nedts
    


def nedt_wl(nedrs, radiances, wlens):
    """
    wavelength units: um
    radiance units: W/(m2 sr um)
    """
    c1 = 1.e24 * 2 * _h * _c**2  # W / (m2 sr um-4)
    c2 = 1.e6 * _h * _c / _k_B  # K um
    
    c1_factor = np.divide(c1, wlens**5 * radiances)

    nedts = np.divide(np.divide(c2, wlens),
                      np.log(1 + c1_factor)**2) * np.divide(c1_factor,
                                                            radiances * (1 + c1_factor)) * nedrs
    
    return nedts



def radiance_offset(ch_y, ch_x, source_y, source_x):
    """
    Intended to find the offset in the x&y directions of a channel at ch_spec with radiance ch_rad
    from a predetermined radiance source function rad_source_fn sampled along spec_basis
    Used as a diagnostic to find which channels showing radiance differences from blackbody,
    and will only work with a smooth radiance curve.
    Tolerance variable might also need tweaking - use only in testing with caution!
    """
    try:
        # y-offset
        ch_x_loc = np.argmin(np.abs(source_x - ch_x))
        source_y_at_ch = source_y[ch_x_loc]
        y_offset = ch_y - source_y_at_ch

        # x-offset
        tolerance = 0.001
        close_y_locs = np.argwhere(np.logical_and((1-tolerance)*ch_y < source_y, source_y < (1+tolerance)*ch_y))
        closest_y_point = close_y_locs[np.argmin(np.abs(ch_x_loc - close_y_locs))][0]
        x_offset = ch_x - source_x[closest_y_point]

        return x_offset, y_offset
    except ValueError:
        print(f'issue occurred')
        pass


def radiance_integral(radiances, spec_basis, srf=1., pf=1.):
    """
    Integrates radiances along spec_basis to convert to radiance (flux per solid angle) from spectral radiance (+ per spectral unit)
    srf: spectral response function also sampled at spec_basis, or =1 if none given
    pf: prefactor (e.g. for conversion from W to mW)
    """
    integrated_radiance = pf * trapezoid(srf * radiances, spec_basis)

    return integrated_radiance



