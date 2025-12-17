"""
Functions for use in working with observational instruments.

Author: Ben Pery
"""

import numpy as np
import xarray as xr
from . import planck, util
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft

def apply_srf(radiance, spec_basis, srf):
    """
    Applies spectral response function for a radiometer-type instrument with spectral basis v:
        channel radiance = integral( srf(v) * radiance(v) )dv / integral srf(v) dv
    """

    integrated_radiance = trapezoid(srf * radiance, spec_basis)
    normalisation = trapezoid(srf, spec_basis)
    channel_radiance = np.divide(integrated_radiance, normalisation)

    return channel_radiance


def prefire_simulation(spec, wn, inst, spec_range=(100,1600), sc=0, reject_mask=0b000011, verbose=False):
    """
    Applies PREFIRE TIRS1/2 spectral response functions to a high-resolution atmospheric radiance simulation.

    spec        : spectrum of input simulation, in units of W / (cm2 sr cm-1)-1
    wn          : wavenumbers of the input simulation, in units of cm-1
    inst        : string 'TIRS1' or 'TIRS2' as appropriate
    spec_range  : spectral range (units cm-1) of input simulation (SRFs not applied for channels outs of this range)
    sc          : int (0-7) indicating which sensor array to simulate (SRFs should be identical across sc, but 
                channels with noise or quality flag issues will be masked)
    reject_mask : binary mask indicating which flags to reject (1) or let pass (0) when calculating SRFs and noise.
                see code below to determine which bits to flag
    verbose     : boolean option for outputting more information about each channel as scanned

    output_ds   : xarray Dataset object containing variables channel_radiance and NEDR_wn, added to
                noise characteristics original to the SRF file.
    """

    # QC filtering options
    flag_descriptors = ['is masked',                                                    # b0
                        'has extreme noise or is unresponsive',                         # b1
                        'is in greater-noise category',                                 # b2
                        'has unreliable calibration due to stray light contribution',   # b3
                        'has unreliable calibration due to thermal effects',            # b4
                        'has unreliable calibration due to filter edge effects']        # b5
    # bit numbers    b543210 
    # reject_mask = 0b000011

    srf_filepath = '/net/thunder/users/bjp224/PREFIRE/SRFs/PREFIRE_' + inst + '_SRF_v13_2024-09-15.nc'

    tirs_srf = xr.load_dataset(srf_filepath)
    n_ch = tirs_srf.channel.size

    # quality filtering, rejecting channels & scenes with specified flags
    channel_masks = tirs_srf['detector_bitflags'][:,sc].data.astype(int)
    channels_to_reject_flagged = np.where(channel_masks & reject_mask)[0]

    # verbose output for channel-by-channel diagnostics
    if verbose:

        bits = 2**np.arange(6)
        bitflag_dict = dict(zip(bits,flag_descriptors))

        for ch, mask in enumerate(channel_masks):

            has_flags = mask > 0

            if has_flags:
                reject_channel = mask & reject_mask

                flags = [mask & 2**bit for bit in range(6)]
                print(f'Channel {ch} detector ' + ', '.join([bitflag_dict[flag] for flag in flags if flag != 0]) + '.')
                if reject_channel:
                    print(f'Channel {ch} excluded.')


    # wavenumber filtering, rejecting channels which lie outside of simulation range
    channels_to_reject_spectral = np.where(   # could just use an OR here 
        np.logical_not(
            np.logical_and(spec_range[0] < tirs_srf['channel_wavenum1'][:,sc].data,
                        tirs_srf['channel_wavenum2'][:,sc].data < spec_range[1])
                        ))[0]

    # combining QC and wavenumber filtering to define rejection list
    channels_to_reject = list(set(np.append(channels_to_reject_flagged, channels_to_reject_spectral)))
    channels_to_reject.sort()

    # creating wavenumber spectrum and selecting spectral range
    wnums_tirs = 1.e4 / tirs_srf.wavelen.data  # conversion of spectrum in µm to cm-1
    longwave_selection = util.in_range(wnums_tirs, spec_range)
    wnums_tirs = wnums_tirs[longwave_selection][::-1]

    # calculating TIRS NEDR conversion from W/(m2 sr µm) to mW/(m2 sr cm-1)
    NEDR_wnum = 1.e3 * np.divide(tirs_srf.channel_mean_wavelen.data, tirs_srf.channel_mean_wavenum.data) * tirs_srf.NEDR

    # variables for saving later from SRF
    srf_vars = [var for var in tirs_srf.variables]
    vars_to_keep = ['channel_wavelen1',
                    'channel_wavelen2',
                    'channel_wavenum1',
                    'channel_wavenum2',
                    'channel_center_wavelen',
                    'channel_center_wavenum',
                    'channel_mean_wavelen',
                    'channel_mean_wavenum',
                    'detector_bitflags',
                    'NEDR',]
    vars_to_drop = [var for var in srf_vars if var not in vars_to_keep]
    vars_1d = ['channel_wavelen1',
                'channel_wavelen2',
                'channel_wavenum1',
                'channel_wavenum2',
                'channel_center_wavelen',
                'channel_center_wavenum',
                'channel_mean_wavelen',
                'channel_mean_wavenum',]

    # make correctly-sized arrays for radiative transfer results
    rt_spec_range = util.in_range(wn, (np.max([np.min(wnums_tirs),spec_range[0]]), spec_range[1]))
    wnums_radtran = wn[rt_spec_range]
    radiances_radtran = 1.e7 * spec[rt_spec_range]  # conversion of radiance from W/cm2 to mW/m2

    # loop over channels to calculate radiances
    channel_radiances = np.zeros(n_ch)
    for ch in range(n_ch):
        if ch not in channels_to_reject:
            srf_radtran_res = np.interp(wnums_radtran, wnums_tirs, tirs_srf['srf'][:,ch,sc].data[longwave_selection][::-1])
            channel_radiances[ch] = apply_srf(radiances_radtran, wnums_radtran, srf_radtran_res)

    # Dataset to return as netcdf
    output_ds = tirs_srf.copy().drop_vars(vars_to_drop)
    for var_1d in vars_1d:
        output_ds[var_1d] = output_ds[var_1d].sel(scene=0, drop=True)
    
    # storing variables
    output_ds['channel_radiance'] = (['channel',], channel_radiances)
    output_ds['channel_radiance'] = output_ds.channel_radiance.assign_attrs({
        'units':'mW / (m2 sr cm-1)',
        'long_name':'Simulated spectral radiance at channel',
    })

    output_ds['NEDR_wn'] = (['channel', 'scene',], NEDR_wnum.data)
    output_ds['NEDR_wn'] = output_ds.NEDR_wn.assign_attrs({
        'units':'mW / (m2 sr cm-1)',
        'long_name':'Noise equivalent delta radiance, converted'
    })

    # assigning information attributes to the dataset
    output_ds = output_ds.assign_attrs({
        'description':'Simulations of PREFIRE radiances from LBLRTM calculations',
        'SRF_distributed_file':'PREFIRE_' + inst + '_SRF_v13_2024-09-15.nc',
    })

    return output_ds



def apply_ils(spec, wn, rt_res=0.001, inst_res=0.3, unit_conv=1.e7, spec_range=(100,1600), lineshape_file='/home/bjp224/FORUM/FORUM_RTTOV_ISRF.txt', calc_brightness_temp=True):
    """
    Applies effect of instrument line shape to a spectrum by convolving in spectral space.
    Default is to apply FORUM ILS.
    unit_conv is multiplicative to get final output in mW/(m2 sr cm-1) units.
    """
    # storing radiative transfer results into Dataset object, resampling at rt_res spectral resolution
    radtran = xr.Dataset(
        {'radiance':( ('wn'), spec)},
        coords = {"wn" : wn}
    )
    # if set to None, then calculates based off source difference between wavenumbers (should be constant)
    if rt_res == None:
        rt_res = radtran.wn.diff(dim='wn').mean()
    radtran = radtran.interp(wn=np.arange(*spec_range, rt_res))

    # loading ILS wavenumbers and interferogram (ifg) shape into Dataset, and interpolating to radtran resolution
    ils_wn, ils_ifg = np.loadtxt(lineshape_file, unpack = True)
    ils_range = (np.min(ils_wn),np.max(ils_wn))
    ils = xr.Dataset(
        {'ifg':( ('wn'), ils_ifg)},
        coords = {"wn" : ils_wn}
    ).interp(wn=np.arange(*ils_range, rt_res))

    # performing convolution and dividing by sum of interferogram values, and applying unit conversion
    norm_factor = ils.ifg.values.sum()
    convolved_spectrum = np.convolve(ils.ifg.values, radtran.radiance.values, mode='same') / norm_factor
    simulated_spectrum = unit_conv * convolved_spectrum

    # storing information to return as a Dataset
    simulation = xr.Dataset(
        {'radiance':( ('wn'), simulated_spectrum)},
        coords = {"wn" : radtran.wn.values}
    ).interp(wn=np.arange(*spec_range, inst_res))

    # storing calculated variables
    simulation['radiance'] = (simulation.radiance).assign_attrs({
        'units':'mW/(m2 sr cm-1)',
        'long_name':'Spectral radiance',
    })
    simulation['wn'] = simulation.wn.assign_attrs({
        'units':'cm-1',
        'long_name':'Wavenumber',
    })

    if calc_brightness_temp:
        # calculating simulated brightness temperatures, and storing these too
        sim_bt = planck.inverse_wn(simulation.radiance.values, simulation.wn.values, units_in='mW')
        simulation['br_temp'] = (['wn',],sim_bt)
        simulation['br_temp'] = simulation.br_temp.assign_attrs({
            'units':'K',
            'long_name':'Brightness temperature',
        })

    return simulation


def apply_ils_bins(spec, wn, rt_res=0.001, inst_res=0.5, unit_conv=1.e7, spec_range=(350,1600), ils_file='/users/bjp224/FINESSE/EM27_ILS_WHAFFFERS.nc', calc_brightness_temp=True):
    """
    Applies effect of spectrally-varying instrument line shape, sorted into bins, to a spectrum by convolving in spectral space.
    Default is to apply FINESSE ILS.

    :args:
    :spec:          array-like; simulated high-resolution spectrum to which to apply instrument line shape
    :wn:            array-like; wavenumber grid of input simulated spectrum (same length as spec)
    :rt_res:        float or Nonetype; if given and not None, the input spectrum is interpolated onto a wavenumber grid of constant separation rt_res
    :inst_res:      float; output resolution (i.e. spectral sampling) of simulated spectrum
    :unit_conv:     float; multiplicative factor applied to give output in units of mW/(m2 sr cm-1). Default is for LBLRTM units of W/(cm2 sr cm-1).
    :spec_range:    tuple; spectral range of instrument
    :ils_file:      string; filepath to netCDF Instrument line shape dataset
    :calc_brightness_temp:      bool; whether to calculate brightness temperature values and include in the outputs

    :outputs:
    :simulation:    xarray.Dataset; dataset containing the full spectrum with instrument characteristics
    """
    # storing radiative transfer results into xarray.Dataset object, resampling at rt_res spectral resolution
    radtran = xr.Dataset(
        {'radiance':( ('wn'), spec)},
        coords = {"wn" : wn}
    )
    # if set to None, then calculates based off source difference between wavenumbers (should be constant)
    if rt_res == None:
        rt_res = radtran.wn.diff(dim='wn').mean()
    radtran = radtran.interp(wn=np.arange(*spec_range, rt_res), kwargs={'fill_value':0.0})

    # loading ILS Dataset (ds) from NetCDF and interpolating to radtran spectral resolution
    ds_ils = xr.load_dataset(ils_file)
    ils_range = (np.min(ds_ils['wn']), np.max(ds_ils['wn']))
    ds_ils = ds_ils.interp(wn=np.arange(*ils_range, rt_res))

    # creating array to store instrument simulation
    lbl_radiance = radtran.radiance.values
    lbl_wns = radtran.wn.values
    simulated_spectrum = np.zeros_like(lbl_radiance)
    pad = 10 # cm-1
    # iterate over bins
    for spec_bin in ds_ils['bin'].values:
        lo_bound = ds_ils['lo_bound'][spec_bin].values
        hi_bound = ds_ils['hi_bound'][spec_bin].values
        # this if/else statement prevents convolution being applied for bins outside of the range of the simultation
        if lo_bound < np.max(lbl_wns) and hi_bound > np.min(lbl_wns):
            # boolean mask for whether basis spectrum is between the bounds of the bins, with pad
            bin_mask_wide = util.in_range(lbl_wns, (lo_bound - pad, hi_bound + pad))
            # load ILS relevant for the specific bin
            bin_ils = ds_ils['ils'][:, spec_bin].values[::-1] # reverse ILS to match expected behaviour when convolving
            norm_factor = bin_ils.sum()
            # apply ILS by convolving with entire spectrum
            bin_conv_spectrum = np.convolve(bin_ils, lbl_radiance[bin_mask_wide], mode='same') / norm_factor
            bin_sim_spectrum = unit_conv * bin_conv_spectrum

            # store data only for the bin
            bin_mask_trim = util.in_range(lbl_wns[bin_mask_wide], # trimming spectral selection to remove pad
                                            (lo_bound, hi_bound), incl=True)
            bin_mask = util.in_range(lbl_wns, # mask for total spectral range
                                        (lo_bound, hi_bound), incl=True)
            simulated_spectrum[bin_mask] = bin_sim_spectrum[bin_mask_trim]
        else:
            pass
    
    # storing information to return as a Dataset
    simulation = xr.Dataset(
        {'radiance':( ('wn'), simulated_spectrum)},
        coords = {"wn" : radtran.wn.values}
    ).interp(wn=np.arange(*spec_range, inst_res))
    
    simulation['radiance'] = (simulation.radiance).assign_attrs({
        'units':'mW/(m2 sr cm-1)',
        'long_name':'Spectral radiance',
    })
    simulation['wn'] = simulation.wn.assign_attrs({
        'units':'cm-1',
        'long_name':'Wavenumber',
    })

    if calc_brightness_temp:
        # calculating simulated brightness temperatures, and storing these too
        sim_bt = planck.inverse_wn(simulation.radiance.values, simulation.wn.values, units_in='mW')
        simulation['br_temp'] = (['wn',],sim_bt)
        simulation['br_temp'] = simulation.br_temp.assign_attrs({
            'units':'K',
            'long_name':'Brightness temperature',
        })

    return simulation


def process_spectrum_general(
    frequency,
    radiance,
    fre_grid, # 0.2 is used in Sanjee & Sophie's code
    st,
    ed,
    new_pd,
    apodisation_func=False,
    test_delta=False,
):
    """
    Apply the optical path difference to a spectrum using a boxcar function or a triangle apodisation function to a high resolution spectrum.
    Formerly called apodised_spectrum. 
    Name changed to avoid confusion for FINESSE which requires a spectrally depdendent ILS applied.

    Adapted from apodise_spectra_boxcar_v1.pro
    ;
    ; Original Author: J Murray (14-Oct-2020)
    ;
    ; Additional comments by R Bantges
    ; Version 1: Original
    ;
    Params
    ------
    frequency array
        Original wavenumber scale (cm^-1)
    radiance array
        Original spectrum
    fre_grid float
        The frequency of the  output grid for the apodised spectra (cm^-1)
    st float
        Wavenumber to start apodised spectrum (cm^-1)
    ed float
        Wavenumber to end apodised spectrum (cm^-1)
    new_pd float
        Optical path difference i.e. width of boxcar to apodise (cm)
    apodisation_func string
        deafult=False
        Function to use in addition to boxcar to apodise the spectrum
        Options
        -------
        "triangle" - Triangle function, running from 1 at centre of interferogram
        to zero at edge of interferogram
    test_delta bool
        deafult=False
        If True, the spectrum is taken to be a delta function, can be
        used to test the apodisation. This should return the ILS which is a sinc
        function in the case of a boxcar
        If False input spectrum is used

    Returns
    -------
    wn array
        Wavenumber of apodised spectrum (cm^-1)
    radiance array
        Radiance or transmission of apodised spectrum
        (same units as input)

    Author: Sanjeevani Panditharatne and Laura Warwick
    """
    # Determine the number of samples making up the output spectra
    samples = int(np.round((ed - st) / fre_grid))

    # Define the wavenumber grid resolution (Fixed high resolution grid.
    # The Monochromatic spectra will be interpolated onto this grid for
    # convenience and potentially reduce time taken for the FFT, the arbitrary
    # number of points in the spectra can be such that it significantly slows
    # the FFT.
    # NB: 0.0001 cm-1 was chosen to resolve the spectral features in the
    # high resolution simulation
    dum_new_res = 0.0001
    dum_samples = int(np.round((ed - st) / dum_new_res))
    # The number of samples in the high res frequency scale

    # ********** Define the arrays for the re-interpolated radiance files **********
    # generate a wavenumber scale running from st - ed wavenumbers
    # at new_res cm-1
    new_fre = np.arange(st, ed, fre_grid)
    # generate a wavenumber scale running from st - ed wavenumbers at 0.001 cm-1
    dum_new_fre = np.arange(st, ed, dum_new_res)
    # ******************************************************************************

    # ********** Interpolate the high res radiance to new array scales **********
    f_dum_spec = interp1d(frequency, radiance)
    dum_spec = f_dum_spec(dum_new_fre)
    if test_delta:
        dum_spec = np.zeros_like(
            dum_spec
        )  # These can be set to produce a delta function to check the sinc
        dum_spec[int(15000000 / 2) : int(15000000 / 2) + 101] = 100.0
    # *****************************************************************************

    # FFT the interpolated LBLRTM spectrum
    int_2 = fft(dum_spec)
    # sampling=1./(2*0.01)/samples/100.   # Sampling interval of the interferogram in cm these are the same for the 0.001 and 0.01 spectra
    sampling = 1.0 / (2 * fre_grid) / samples / 100.0
    # Sampling interval of the interferogram in cm these are the same for the 0.001 and 0.01 spectra

    # ********** Apodise the LBLRTM sim and transform **********
    Q = int(
        round(new_pd / 100.0 / sampling / 2.0)
    )  # number of samples required to extend the path difference to 1.26cm
    # *****************************************************************************

    # Define an array to hold the folded out inteferogram
    int_1 = np.zeros(samples, dtype=np.cdouble)

    # 'int_2' - this interferogram is equivalent to a sampling grid of 0.001 cm-1
    # in the spectral domain, this statement applies a boxcar apodisation over +/-1.26 cm
    int_2[Q:-Q] = 0.0

    # The following two lines reduce the output spectra to a sampling grid of 0.01 cm-1
    # while copying in the truncated interferogram from the high resolution interferogram
    int_1[0 : int(round((samples / 2)))] = int_2[0 : int(round((samples / 2)))]
    int_1[int(round((samples / 2))) : samples] = int_2[
        (dum_samples) - int(round((samples / 2))) : dum_samples
    ]

    if apodisation_func == "triangle":
        print("Applying triangle")
        # int_1_unapodised = np.copy(int_1)
        triangle_left = [1, 0]
        triangle_left_x = [0, Q]
        triangle_left_x_all = np.arange(len(int_1[0:Q]) + 1)
        f_triangle_left = interp1d(triangle_left_x, triangle_left)
        triangle_right = [0, 1]
        triangle_right_x = [len(int_1) - Q - 1, len(int_1)]
        triangle_right_x_all = np.arange(len(int_1) - Q - 1, len(int_1), 1)
        f_triangle_right = interp1d(triangle_right_x, triangle_right)

        int_1[0 : Q + 1] = int_1[0 : Q + 1] * f_triangle_left(triangle_left_x_all)
        int_1[-Q - 2 : -1] = int_1[-Q - 2 : -1] * f_triangle_right(triangle_right_x_all)

    elif not apodisation_func:
        print("Applying boxcar")

    else:
        print("No recognised function selected, defaulting to boxcar")

    new_lbl_spec = ifft(int_1)

    # ***********************************************************************
    apodised_spectra = np.real(new_lbl_spec / (fre_grid / dum_new_res))
    return new_fre, apodised_spectra


def apply_inst_function(wn, spec, instrument, unit_conv=1.e7, sc=None):

    if instrument == 'FINESSE':

        wn_FINESSE, radiance_FINESSE = process_spectrum_general(wn, spec, 0.01, 370, 1610, 1.21)
        result = apply_ils_bins(radiance_FINESSE, wn_FINESSE, inst_res=0.2, unit_conv=unit_conv,
                                          spec_range=(352.4,1600.3), calc_brightness_temp=False)
        # result = apply_ils_bins(spec, wn, inst_res=0.2, unit_conv=unit_conv, calc_brightness_temp=False)
        output_wn = result['wn'].values
        output_spec = result['radiance'].values

    elif instrument == 'FORUM':

        result = apply_ils(spec, wn, unit_conv=unit_conv, calc_brightness_temp=False)
        output_wn = result['wn'].values
        output_spec = result['radiance'].values

    elif instrument == 'TIRS1' or instrument == 'TIRS2':
        result = prefire_simulation(spec, wn, instrument, spec_range=(100,1800), sc=sc, reject_mask=0b000011)
        output_wn = result['channel_mean_wavenum'].values
        output_spec = result['channel_radiance'].values

    return output_wn, output_spec