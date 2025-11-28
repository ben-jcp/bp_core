import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
from . import ceda_io, data, util

g_0 = 9.80665  # m s-2, standard gravity
R_e = 6371.e3  # m, mean earth radius
R_star = 287.052874  # J kg-1 K-1, dry air specific gas constant
mass_ratio = 18. / 28.96  # molar mass ratio of H2O to dry air
rho_water = 999.972  # kg m-3, density of liquid water

def profile_from_era(dataset):
    # input: xarray from ERA5 dataset containing data from one latlon grid point
    # would be good for multiple grid points to work

    profile = dataset[['o3','q','t']]

    profile['o3'] = 1.e3 * profile.o3  # kg/kg to g/kg
    profile['o3'] = profile.o3.assign_attrs(
        data.extract_attrs(dataset, 'o3', replace={'units':'g kg**-1'}))

    profile['q'] = 1.e3 * np.divide(profile.q, 1 - profile.q)  # spec hum (kg/kg) to MMR (g/kg)
    profile['q'] = profile.q.assign_attrs(
        data.extract_attrs(dataset, 'q', replace={'units':'g kg**-1',
                                             'long_name':'Water vapour mass mixing ratio','standard_name':'humidity_mixing_ratio'}))
    
    profile['t'] = profile.t.assign_attrs(
        data.extract_attrs(dataset, 't')  # loses GRIB values for easier reading of attrs
    )

    geopot = dataset['z']  # m2 s-2
    profile = profile.assign(z=convert_geopotential_to_altitude(geopot),)
    profile['z'] = profile.z.assign_attrs(
        units='km',
        long_name='Altitude',
        standard_name='altitude',
    )

    try:
        profile['cc'] = dataset['cc'].assign_attrs(
            data.extract_attrs(dataset, 'cc'))
    except KeyError:
        print('no cloud cover data available in ERA5')
        
    print('Profile calculated')

    # 04 Nov 24 WARNING: either assignment of attributes or multiplication
    # overwrites existing attributes: low priority metadata issue

    return profile


def perturb_profile(profile, var_key, pert_shape, pert_mag, pert_extent, extent_key='z'):
    """
    profile:        xarray.Dataset with atmospheric profile
    var_key:        Dataset key for variable to perturb
    pert_shape:     string, options for perturbation type
    pert_mag:       float (fractional magnitude)
    pert_extent:    tuple (layer heights between which to perturb)
    extent_key:     Dataset key for variable to specify layer heights
        NB Needs modifying to work in pressure coordinates (decreasing with altitude)
    """
    shape_info = {'const':'constant fractional perturbation',
                  'lin_lo':'linear fractional perturbation, 0 at top layer, max fraction at bottom layer',
                  'lin_hi':'linear fractional perturbation, 0 at bottom layer, max fraction at top layer'}

    pert_lvls = ((np.min(pert_extent) < profile[extent_key].data)
                 & (profile[extent_key].data < np.max(pert_extent)))

    if pert_shape == 'const':
        pert_fracs = 1 + pert_mag * pert_lvls

    elif pert_shape == 'lin_lo':
        pert_fracs = 1 + pert_mag * pert_lvls * (np.max(pert_extent) - profile[extent_key].data) / np.diff(pert_extent)

    elif pert_shape == 'lin_hi':
        pert_fracs = 1 + pert_mag * pert_lvls * (profile[extent_key].data - np.min(pert_extent)) / np.diff(pert_extent)

    else:
        raise Exception('invalid specification of profile perturbation shape pert_shape')

    new_profile = profile.copy()
    new_profile[var_key] = profile[var_key] * pert_fracs
    new_profile[var_key] = new_profile[var_key].assign_attrs(data.extract_attrs(profile, var_key))

    whole_ds_attrs = new_profile.attrs
    perturb_info = f"Variable {var_key} perturbed by {shape_info[pert_shape]} of {pert_mag:.2f}, in layers between {pert_extent}{profile[extent_key].units}"
    whole_ds_attrs['applied_perturbation'] = perturb_info
    new_profile = xr.Dataset(new_profile, attrs=whole_ds_attrs)

    return new_profile, perturb_info


def convert_geopotential_to_altitude(geopotentials):
    altitude = 1e-3 * np.divide(R_e * geopotentials,
                                g_0 * R_e - geopotentials)  # km
    
    return altitude


def convert_specific_humidity_to_mmr(q):
    # kg/kg spec hum to g/kg mass mixing ratio
    wvmr = 1.e3 * np.divide(q, 1 - q)
    return wvmr


def convert_wvmr_to_specific_humidity(q):
    # g/kg water vapour mass mixing ratio to kg/kg spec hum
    q = 1.e-3 * q
    spec_hum = np.divide(q, 1 + q)
    return spec_hum


def sat_vp_water(T):
    """
    calculates saturation vapour pressure of H2O over liquid water
    after formula from Murphy & Koop (2005, QJRMS)
    valid for 123K < T < 332K
    returns pressures in hPa
    """

    ln_e = (54.842763 
            - 6763.22 / T 
            - 4.210 * np.log(T)
            + 0.000367 * T
            + np.tanh(0.0415 * (T - 218.8))
            * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
    
    e_water = np.exp(ln_e) / 1.e2  # conv

    return e_water


def convert_rh_to_mass_mixing_ratio(rel_hum, pressure, temp):
    """
    Relative humidity (%) conversion to mass mixing ratio (kg/kg)
    Pressure units of hPa
    Tested (in combination with sat_vp_water) on 13-Mar-25
    against https://www.aqua-calc.com/calculate/humidity, agrees within 6 parts in 10 000
    """

    partial_pressure = 1.e-2 * rel_hum * sat_vp_water(temp)

    mmr = mass_ratio * np.divide(partial_pressure, pressure - partial_pressure)

    return mmr


def convert_rh_to_specific_humidity(rel_hum, pressure, temp):
    """
    """
    partial_pressure = 1.e-2 * rel_hum * sat_vp_water(temp)

    q = mass_ratio * np.divide(partial_pressure, pressure - (1 - mass_ratio) * partial_pressure)

    return q


def convert_mass_mixing_ratio_to_rh(mmr, pressure, temp):
    
    mmr = 1.e-3 * mmr # conversion from g/kg to kg/kg
    rh = 1.e2 * pressure * mmr / ( (mmr + mass_ratio) * sat_vp_water(temp))    
    
    return rh


def half_lvls_ecmwf(lnsp):

    # load in ECMWF coefficients for calculating model levels: taken from https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
    coeff_fp = '/net/thunder/data1/bjp224/ecmwf_data/ecmwf_model_level_defs.csv'
    n_half_lvl, a_coeffs, b_coeffs = np.loadtxt(coeff_fp, delimiter=',', skiprows=1).T

    if type(lnsp) == np.ndarray or type(lnsp) == list:
        if len(lnsp) == 1:
            lnsp = lnsp[0]

    # calculation of half-level pressures
    p_half_lvls = a_coeffs + np.multiply.outer(np.exp(lnsp), b_coeffs) # currently in Pa

    return p_half_lvls


def p_lvls_ecmwf(lnsp):

    # # load in ECMWF coefficients for calculating model levels: taken from https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
    # coeff_fp = '/net/thunder/data1/bjp224/ecmwf_data/ecmwf_model_level_defs.csv'
    # n_half_lvl, a_coeffs, b_coeffs = np.loadtxt(coeff_fp, delimiter=',', skiprows=1).T

    # if type(lnsp) == np.ndarray or type(lnsp) == list:
    #     if len(lnsp) == 1:
    #         lnsp = lnsp[0]

    # # calculation of half-level pressures
    # p_half_lvls = a_coeffs + np.multiply.outer(np.exp(lnsp), b_coeffs) # currently in Pa

    p_half_lvls = half_lvls_ecmwf(lnsp)

    # conversion to full-level pressures in hPa, by averaging consecutive entries
    # this is done with the convolution operator, applied along the vertical axis
    p_full_lvls = 1.e-2 * np.apply_along_axis(np.convolve, axis=-1, arr=p_half_lvls, v=[0.5, 0.5], mode="valid")

    return p_full_lvls


def profile_from_ml_era5(datetime_str, point=None, loc_range=None):
    
    # checking for specification of location selection
    if point and loc_range:
        raise Exception('Cannot specify both a spatial range and a single point')
    elif point:
        filt_params = point
    elif loc_range:
        filt_params = loc_range

    # generating filepaths for dataset at this time/date
    source_fp_ml = util.ceda_era_fp(datetime_str, 'an_ml')
    source_fp_sfc = util.ceda_era_fp(datetime_str, 'an_sfc')

    # getting token for CEDA access
    token, expires = ceda_io.get_token()

    # load lnsp dataset
    lnsp_ds = ceda_io.open_ceda_dataset(source_fp_ml('lnsp'), token=token)
    lnsp_ds = data.spatial_filter(lnsp_ds, filt_params)
    
    # reading in time, lat, lon
    time_data = lnsp_ds.time.values
    lat_data = lnsp_ds.latitude.values
    lon_data = lnsp_ds.longitude.values

    # nesting of data in a list required to make xarray happy
    if lat_data.ndim == 0:
        lat_data = [lat_data]
        lon_data = [lon_data]

    # generate pressure levels from lnsp value(s)
    p_hl_data = 1.e-2 * half_lvls_ecmwf(lnsp_ds.lnsp.values)
    p_fl_data = p_lvls_ecmwf(lnsp_ds.lnsp.values)

    lnsp_ds.close()
    
    # get size of dataset and broadcast pressure levels into that shape
    n_lats_gp = len(lat_data)
    n_lons_gp = len(lon_data)
    n_lvls = 137
    p_fl_data = np.broadcast_to(p_fl_data, (n_lats_gp, n_lons_gp, n_lvls))
    p_hl_data = np.broadcast_to(p_hl_data, (n_lats_gp, n_lons_gp, n_lvls+1))

    # start local dataset, using model level as coordinate and pressure as variable
    profile_ds = xr.Dataset(
        data_vars={
        'p':(['latitude','longitude','level'],p_fl_data),
        'p_half':(['latitude','longitude','half_level'],p_hl_data),
        'time':(time_data),
        },
        coords={
        'latitude':(lat_data),
        'longitude':(lon_data),    
        }
    )

    # load in T, q, O3, and geopotential
    for var_ml in ['t', 'q', 'o3', 'z']:
        ds = ceda_io.open_ceda_dataset(source_fp_ml(var_ml), token=token)
        ds = data.spatial_filter(ds, filt_params)

        if var_ml != 'z':

            var_data = ds[var_ml].values[0]

            if point:
                # reallocates data to have correct structure even if just 1 gridpoint chosen
                var_data = var_data[:, np.newaxis, np.newaxis]

            # assign values of variable to dataset, and reorder dimensions
            profile_ds = profile_ds.assign({var_ml:(['level','latitude','longitude'],var_data)})
            profile_ds[var_ml] = profile_ds[var_ml].transpose('latitude','longitude','level')

        else:
            # special case for geopotential which is a 2D field specified at the surface only
            var_data = np.broadcast_to(ds[var_ml].values[0], (n_lats_gp, n_lons_gp,))
            profile_ds = profile_ds.assign({var_ml:(['latitude','longitude',],var_data)})

        ds.close()

    # dealing with skin temp separately as it is defined for single-level, not model level
    skt_ds = ceda_io.open_ceda_dataset(source_fp_sfc('skt'), token=token)
    skt_ds = data.spatial_filter(skt_ds, filt_params)

    skt_data = np.broadcast_to(skt_ds['skt'].values[0], (n_lats_gp, n_lons_gp,))
    profile_ds = profile_ds.assign({'skt':(['latitude','longitude',],skt_data)})

    skt_ds.close()

    # unit conversions
    profile_ds['o3'] = 1.e3 * profile_ds['o3']  # kg/kg to g/kg
    profile_ds['q'] = convert_specific_humidity_to_mmr(profile_ds['q'])  # spec hum kg/kg to mass mixing ratio g/kg
    profile_ds['z'] = convert_geopotential_to_altitude(profile_ds['z'])  # geopotential to altitude (km)

    # change longitude to 'natural' i.e. (-180,180)
    profile_ds['longitude'] = np.mod(profile_ds['longitude'] + 180, 360)-180
    profile_ds = profile_ds.sortby('longitude')

    # add in units and other attributes to all variables
    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        'p_half':{'units':'hPa', 'long_name':'half_level_pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'q':    {'units':'g kg**-1', 'long_name':'water vapour mass mixing ratio', 'standard_name':'humidity_mixing_ratio'},
        'o3':   {'units':'g kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'z':    {'units':'km', 'long_name':'surface altitude', 'standard_name':'surface_altitude'},
        'skt':  {'units':'K', 'long_name':'skin temperature',},
        'latitude': {'units':'degrees_north', 'long_name':'latitude',},
        'longitude':{'units':'degrees_east', 'long_name':'longitude',},
        'level':    {'long_name':'model_level_number',},
        'time':     {'long_name':'time',},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    print('ERA5 profile(s) loaded successfully')

    return profile_ds


def hypsometric_altitudes(pressure_fl, pressure_hl, temperature, wv_mmr, sfc_altitude, half_lvls=False):
    # takes profiles sorted by decreasing pressure i.e. increasing altitude
    # returns altitudes for each level in kilometres 

    if pressure_hl.ndim == 1:
        horiz_shape = 1
    elif pressure_hl.ndim > 1:
        # assumes that if input is >1D, that pressure levels are the last coordinate
        horiz_shape = (*pressure_hl.shape[:-1], 1)
        sfc_altitude = sfc_altitude.reshape((*sfc_altitude.shape,1))

    # spec_hum = profile.q.values / (1 + profile.q.values)
    spec_hum = convert_wvmr_to_specific_humidity(wv_mmr)
    t_virtual = temperature * (1 + (-1 + 1/mass_ratio) * spec_hum)
    # hydrostatic relation to calculate approx ∆z = (R* T_v / p g) * ∆p. conversion to km 
    layer_thicknesses = - 1.e-3 * R_star * t_virtual * np.diff(pressure_hl,) / (pressure_fl * g_0)
    altitudes_hl = np.append(np.zeros(horiz_shape), np.cumsum(layer_thicknesses, axis=-1), axis=-1) + sfc_altitude
    altitudes = np.apply_along_axis(np.convolve, axis=-1, arr=altitudes_hl, v=[0.5, 0.5], mode="valid")

    if half_lvls:
        return altitudes, altitudes_hl
    else:
        return altitudes


def add_altitudes_era(profile):
    # add calculated altitudes to standard ERA5 model level profile

    # if pressures ascending, sort to have descending instead, for hypsometric calculation
    # maybe should have this done automatically in hypsometric_altitudes()
    if np.all( np.diff(profile['p'].values) > 0 ):
        profile = profile.sortby(['p','p_half'], ascending=False)

    altitudes, altitudes_hl = hypsometric_altitudes(profile.p.values, profile.p_half.values,
                                                    profile.t.values, profile.q.values,
                                                    profile.z.values, half_lvls=True)

    profile = profile.assign({'altitude':(list(profile.p.dims), altitudes),
                              'altitude_half':(list(profile.p_half.dims), altitudes_hl)})
    profile = data.add_units(profile, {'altitude':'km', 'altitude_half':'km'})

    return profile


def profile_from_ml_era5_new(datetime_str, point=None, loc_range=None, load_local=True, attempt_download=False, tag=None, dl_area=None):
    
    # checking for specification of location selection
    if point and loc_range:
        raise Exception('Cannot specify both a spatial range and a single point')
    elif point:
        filt_params = point
    elif loc_range:
        filt_params = loc_range

    # 
    if load_local:
        time_label = 'valid_time'
        level_label = 'model_level'
        if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
            load_source = 'disk'
        else:
            if attempt_download:
                dt_obj = dt.datetime.fromisoformat(datetime_str)
                dl_date = dt_obj.strftime('%Y-%m-%d')
                hr_pad = 2
                dl_times = np.arange(dt_obj.hour-hr_pad, dt_obj.hour+hr_pad+1)
                dl_times = dl_times[util.in_range(dl_times, (0,23), incl=True)]
                
                print(f'Attempting download for {dl_date}: {len(dl_times)} times selected')
                data.download_era5_ml(['lnsp','t','q','o3','z','skt','cc'],
                                      dl_date, dl_times, 'all', 'oper', 'an',
                                      area_NWSE=dl_area, tag=tag)
                # TODO make to download at profile date, times ± 2h (within day)
                # area and tag supplied externally
                if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
                    load_source = 'disk'
                else:
                    raise FileNotFoundError(f'Download attempted but complete ERA5 data not found on disk for {datetime_str}')
            else:
                raise FileNotFoundError(f'ERA5 data not found on disk for {datetime_str} and local download not attempted')
    else:
        time_label = 'time'
        level_label = 'level'
        load_source = 'ceda'

    # generating filepaths for dataset at this time/date
    source_fp_ml = util.any_era_fp(datetime_str, 'an_ml', tag, source=load_source)
    source_fp_sfc = util.any_era_fp(datetime_str, 'an_sfc', tag, source=load_source)

    # getting token for CEDA access
    if load_source == 'ceda':
        token, expires = ceda_io.get_token()
    else:
        token = ''

    # load lnsp dataset
    lnsp_ds = data.dataset_load_and_process(source_fp_ml('lnsp'), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)
    
    # reading in time, lat, lon
    time_data = lnsp_ds[time_label].values
    lat_data = lnsp_ds.latitude.values
    lon_data = lnsp_ds.longitude.values

    # nesting of data in a list required to make xarray happy
    if lat_data.ndim == 0:
        lat_data = [lat_data]
        lon_data = [lon_data]

    # generate pressure levels from lnsp value(s)
    p_hl_data = 1.e-2 * half_lvls_ecmwf(lnsp_ds.lnsp.values)
    p_fl_data = p_lvls_ecmwf(lnsp_ds.lnsp.values)

    lnsp_ds.close()
    
    # get size of dataset and broadcast pressure levels into that shape
    n_lats_gp = len(lat_data)
    n_lons_gp = len(lon_data)
    n_lvls = 137
    p_fl_data = np.broadcast_to(p_fl_data, (n_lats_gp, n_lons_gp, n_lvls))
    p_hl_data = np.broadcast_to(p_hl_data, (n_lats_gp, n_lons_gp, n_lvls+1))

    # initiate local dataset, using model level as coordinate and pressure as variable
    profile_ds = xr.Dataset(
        data_vars={
        'p':(['latitude','longitude','level'],p_fl_data),
        'p_half':(['latitude','longitude','half_level'],p_hl_data),
        'time':(time_data),
        },
        coords={
        'latitude':(lat_data),
        'longitude':(lon_data),    
        }
    )

    # load in T, q, O3, and geopotential
    for var_ml in ['t', 'q', 'o3', 'z']:
        ds = data.dataset_load_and_process(source_fp_ml(var_ml), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)

        if var_ml != 'z':

            var_data = ds[var_ml].values[0]

            if point:
                # reallocates data to have correct structure even if just 1 gridpoint chosen
                var_data = var_data[:, np.newaxis, np.newaxis]

            # assign values of variable to dataset, and reorder dimensions
            profile_ds = profile_ds.assign({var_ml:(['level','latitude','longitude'],var_data)})
            profile_ds[var_ml] = profile_ds[var_ml].transpose('latitude','longitude','level')

        else:
            # special case for geopotential which is a 2D field specified at the surface only
            var_data = np.broadcast_to(ds[var_ml].values[0], (n_lats_gp, n_lons_gp,))
            profile_ds = profile_ds.assign({var_ml:(['latitude','longitude',],var_data)})

        ds.close()

    # dealing with skin temp separately as it is defined for single-level, not model level
    skt_ds = data.dataset_load_and_process(source_fp_sfc('skt'), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)
    
    skt_data = np.broadcast_to(skt_ds['skt'].values[0], (n_lats_gp, n_lons_gp,))
    profile_ds = profile_ds.assign({'skt':(['latitude','longitude',],skt_data)})

    skt_ds.close()

    # unit conversions
    profile_ds['o3'] = 1.e3 * profile_ds['o3']  # kg/kg to g/kg
    profile_ds['q'] = convert_specific_humidity_to_mmr(profile_ds['q'])  # spec hum kg/kg to mass mixing ratio g/kg
    profile_ds['z'] = convert_geopotential_to_altitude(profile_ds['z'])  # geopotential to altitude (km)

    # change longitude to 'natural' i.e. (-180,180)
    if np.max(profile_ds['longitude']) > 180:
        profile_ds['longitude'] = np.mod(profile_ds['longitude'] + 180, 360)-180
        profile_ds = profile_ds.sortby('longitude')

    # add in units and other attributes to all variables
    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        'p_half':{'units':'hPa', 'long_name':'half_level_pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'q':    {'units':'g kg**-1', 'long_name':'water vapour mass mixing ratio', 'standard_name':'humidity_mixing_ratio'},
        'o3':   {'units':'g kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'z':    {'units':'km', 'long_name':'surface altitude', 'standard_name':'surface_altitude'},
        'skt':  {'units':'K', 'long_name':'skin temperature',},
        'latitude': {'units':'degrees_north', 'long_name':'latitude',},
        'longitude':{'units':'degrees_east', 'long_name':'longitude',},
        'level':    {'long_name':'model_level_number',},
        'time':     {'long_name':'time',},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    print('ERA5 profile(s) loaded successfully')

    return profile_ds


def sonde_era_hybrid(filepath, n_layers, spacing, vertical_coord):

    sonde_data = data.load_radiosonde_nc(filepath)

    vertical_coord = vertical_coord.casefold().strip()
    if vertical_coord in ['pressure', 'p']:
        vc_label = 'PRESSURE_INSITU'
    elif vertical_coord in ['altitude', 'z']:
        vc_label = 'ALTITUDE'

    vert_min, vert_max = sonde_data[vc_label].values.min(), sonde_data[vc_label].values.max()

    p_sonde = sonde_data['PRESSURE_INSITU'].values
    t_sonde = sonde_data['TEMPERATURE_INSITU'].values + 273.15  # convert degC to K
    z_sonde = 1.e-3 * sonde_data['ALTITUDE'].values  # convert m to km
    wv_mmr_sonde = 1.e3 * convert_rh_to_mass_mixing_ratio(sonde_data['HUMIDITY.RELATIVE_INSITU'], p_sonde, t_sonde)
    sonde_time = sonde_data['time'].values
    sonde_lat = sonde_data['LATITUDE'].values
    sonde_lon = sonde_data['LONGITUDE'].values

    # defining levels (i.e. layer boundaries)
    spacing = spacing.casefold().strip()
    if spacing == 'linear' or spacing == 'lin':
        level_heights = np.linspace(vert_min, vert_max, n_layers+1)
    elif spacing == 'log':
        # will work as expected for altitude vert coord, but NB for pressure gives more layers near TOA
        level_heights = np.logspace(np.log10(vert_min), np.log10(vert_max), n_layers+1)
    elif spacing == 'custom_brb' and vc_label == 'PRESSURE_INSITU':
        alpha = 1.45
        level_heights = vert_max - (vert_max - vert_min) * (np.arange(n_layers+1) / n_layers) ** alpha

    p, t, z, wv_mmr, n_readings = np.zeros((5, n_layers,))
    sonde_times_avg, sonde_lats_avg, sonde_lons_avg = np.zeros((3, n_layers,))
    sonde_times_avg = sonde_times_avg.astype('<M8[us]')

    for i in range(n_layers):
        # mask to select sonde datapoints within range of pressures
        vert_coords = sonde_data[vc_label].values
        if vc_label == 'PRESSURE_INSITU':
            level_heights = np.sort(level_heights)[::-1]
            layer_bounds = np.flip(level_heights[i:i+2])
        else:
            layer_bounds = level_heights[i:i+2]

        mask_single_level = util.in_range(vert_coords, layer_bounds, incl=True)

        # averaging over variables
        p_fl = np.average(p_sonde[mask_single_level])
        t_fl = np.average(t_sonde[mask_single_level])
        z_fl = np.average(z_sonde[mask_single_level])
        wv_mmr_fl = np.average(wv_mmr_sonde[mask_single_level])

        n_fl = np.sum(mask_single_level)

        time_avg = pd.to_datetime( np.average(sonde_time[mask_single_level].astype(np.int64)), unit='us')
        # time_avg = np.average(sonde_time[mask_single_level].astype(int)).astype(sonde_time.dtype)
        lat_avg = np.average(sonde_lat[mask_single_level])
        lon_avg = np.average(sonde_lon[mask_single_level])

        p[i], t[i], z[i], wv_mmr[i], n_readings[i] = p_fl, t_fl, z_fl, wv_mmr_fl, n_fl
        sonde_times_avg[i], sonde_lats_avg[i], sonde_lons_avg[i] = time_avg, lat_avg, lon_avg

    sonde_times_avg = [str(time) for time in sonde_times_avg]

    # sonde_times_avg = sonde_times_avg.astype('<M8[ns]')

    # loading in relevant ERA5 for launch time and location
    # would need to rewrite here to have ERA5 following sonde, including check that sonde within data bounding box
    # would also need to add CAMS data at this step
    launch_time = str(sonde_data['time'].values[0])
    launch_coords = (sonde_data['LATITUDE'].values[0], sonde_data['LONGITUDE'].values[0])
    era5_ds = profile_from_ml_era5_new(launch_time, point=launch_coords, attempt_download=True,
                                               tag='eastCanada', dl_area=[48, -77, 43, -70]).squeeze()
    era5_ds = era5_ds.sortby(['p','p_half'], ascending=False)
    era5_ds = add_altitudes_era(era5_ds)

    # load in CAMS data and add to existing profile dataset
    cams_ghg_ds = ghg_profile_cams(launch_time, point=launch_coords, tag='eastCanada')
    era5_ds = merge_ghg_to_profile(era5_ds, cams_ghg_ds)

    era5_time = era5_ds.time.values
    era5_lat = era5_ds.latitude.values
    era5_lon = era5_ds.longitude.values
    
    # FIRST: decide where ERA5 for all variables to be used
            # LATER: add in ERA5 filler for gaps
    if vertical_coord in ['pressure', 'p']:
        era_vc_label = 'p'
        low_range = (vert_max, 1100.)
        high_range = (0., vert_min)
    elif vertical_coord in ['altitude', 'z']:
        vert_min, vert_max = 1.e-3 * vert_min, 1.e-3 * vert_max
        era_vc_label = 'altitude'
        low_range = (0., vert_min - 0.01) # OFFSET ADDED 04-08-25 TO AVOID OVERLAPPING LAYERS when switching between p/z coordinates
        high_range = (vert_max, 100.)

    lo_mask = util.in_range(era5_ds[era_vc_label].values, low_range)
    hi_mask = util.in_range(era5_ds[era_vc_label].values, high_range)

    # SECOND: take ERA5 top and bottom levels as they come
    # could write this as a 'nest arrays' function
    p_both_sources = np.concatenate([era5_ds['p'].values[lo_mask], p, era5_ds['p'].values[hi_mask]])
    t_both_sources = np.concatenate([era5_ds['t'].values[lo_mask], t, era5_ds['t'].values[hi_mask]])
    z_both_sources = np.concatenate([era5_ds['altitude'].values[lo_mask], z, era5_ds['altitude'].values[hi_mask]])
    wv_mmr_both_sources = np.concatenate([era5_ds['q'].values[lo_mask], wv_mmr, era5_ds['q'].values[hi_mask]])

    nat_array = np.full(len(era5_ds.p.values), 'NaT')
    nan_array = np.full_like(era5_ds.p.values, np.nan)
    sonde_times_avg = np.concatenate([nat_array[lo_mask], sonde_times_avg, nat_array[hi_mask]])
    sonde_times_avg = np.asarray(sonde_times_avg, dtype=np.datetime64)
    sonde_lats_avg = np.concatenate([nan_array[lo_mask], sonde_lats_avg, nan_array[hi_mask]])
    sonde_lons_avg = np.concatenate([nan_array[lo_mask], sonde_lons_avg, nan_array[hi_mask]])
    sonde_n_rdgs = np.concatenate([nan_array[lo_mask], n_readings, nan_array[hi_mask]])

    sonde_bounds_p = [sonde_data['PRESSURE_INSITU'].values.max(), sonde_data['PRESSURE_INSITU'].values.min()]
    sonde_bounds_z = [1.e-3*sonde_data['ALTITUDE'].values.min(), 1.e-3*sonde_data['ALTITUDE'].values.max()]

    data_source = np.full(len(p_both_sources), 'Radiosonde iMet4')
    data_source[np.isnan(sonde_times_avg)] = 'ERA5'

    # THIRD: add in skt, and o3 interpolated to vertical levels
    skintemp_era = era5_ds.skt.values
    o3_interp_era, co2_interp, co_interp, ch4_interp = data.nondim_interp(era5_ds, p_both_sources, 'p', ['o3', 'co2', 'co', 'ch4'])

    # PROFILE: things as coord of pressure
    profile_ds = xr.Dataset(
        data_vars={
        't': (['p'], t_both_sources),
        'z': (['p'], z_both_sources),
        'q': (['p'], wv_mmr_both_sources),
        'o3': (['p'], o3_interp_era),
        'co2': (['p'], co2_interp),
        'co': (['p'], co_interp),
        'ch4': (['p'], ch4_interp),
        'skt': (skintemp_era),
        'data_source': (['p'], data_source),
        'sonde_time': (['p'], sonde_times_avg),
        'sonde_latitude': (['p'], sonde_lats_avg),
        'sonde_longitude': (['p'], sonde_lons_avg),
        'sonde_readings': (['p'], sonde_n_rdgs),
        'era5_time': (era5_time),
        'era5_latitude': (era5_lat),
        'era5_longitude': (era5_lon),
        'sonde_bounds_press': (sonde_bounds_p),
        'sonde_bounds_alt': (sonde_bounds_z),
        },
        coords={
        'p':(p_both_sources),
        }
    )

    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'z':    {'units':'km', 'long_name':'geometric altitude', 'standard_name':'altitude'},
        'q':    {'units':'g kg**-1', 'long_name':'water vapour mass mixing ratio', 'standard_name':'humidity_mixing_ratio'},
        'o3':   {'units':'g kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'co2':   {'units':'g kg**-1', 'long_name':'carbon dioxide mass mixing ratio'},
        'co':   {'units':'g kg**-1', 'long_name':'carbon monoxide mass mixing ratio'},
        'ch4':   {'units':'g kg**-1', 'long_name':'methane mass mixing ratio'},
        'skt':  {'units':'K', 'long_name':'skin temperature',},
        'sonde_time':     {'long_name':'radiosonde averaged time',},
        'sonde_latitude': {'units':'degrees_north', 'long_name':'radiosonde averaged latitude',},
        'sonde_longitude':{'units':'degrees_east', 'long_name':'radiosonde averaged longitude',},
        'sonde_readings': {'long_name':'number of radiosonde readings averaged',},
        'era5_time':     {'long_name':'ERA5 profile time',},
        'sonde_bounds_press': {'units':'hPa', 'long_name':'pressure range of radiosonde measurements'},
        'sonde_bounds_alt': {'units':'km', 'long_name':'altitude range of radiosonde measurements'},
        'era5_latitude': {'units':'degrees_north', 'long_name':'ERA5 profile latitude',},
        'era5_longitude':{'units':'degrees_east', 'long_name':'ERA5 profile longitude',},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    info_dict = {'name':'hybrid radiosonde-ERA5 profile',
                 'sonde_processing':
                 f'Radiosonde data averaged on layers defined by levels with spacing {spacing} in coordinate {vertical_coord}',
                 'era_processing':'skt taken directly, o3 interpolated onto pressure layers. all fields in layers outside of radiosonde range defined by ERA5'}

    profile_ds = profile_ds.assign_attrs(info_dict)

    return profile_ds


def ghg_profile_cams(datetime_str, point=None, loc_range=None, load_local=True, attempt_download=False, tag=None, dl_area=None):

    # get co2, co, ch4 as fn of model level for the selected location.
    # convert kg/kg to g/kg for LBLRTM
    # implement auto-download asap if possible
    # searh for closest valid_time for the single forecast_period

    load_source = 'disk'

    # checking for specification of location selection
    if point and loc_range:
        raise Exception('Cannot specify both a spatial range and a single point')
    elif point:
        filt_params = point
    elif loc_range:
        filt_params = loc_range

    token=None

    # 
    # if load_local:
    #     time_label = 'valid_time'
    #     level_label = 'model_level'
    #     if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
    #         load_source = 'disk'
    #     else:
    #         if attempt_download:
    #             dt_obj = dt.datetime.fromisoformat(datetime_str)
    #             dl_date = dt_obj.strftime('%Y-%m-%d')
    #             hr_pad = 2
    #             dl_times = np.arange(dt_obj.hour-hr_pad, dt_obj.hour+hr_pad)
    #             dl_times = dl_times[util.in_range(dl_times, (0,23), incl=True)]
                
    #             print(f'Attempting download for {dl_date}: {len(dl_times)} times and ')
    #             data.download_era5_ml(['lnsp','t','q','o3','z','skt','cc'],
    #                                   dl_date, dl_times, 'all', 'oper', 'an',
    #                                   area_NWSE=dl_area, tag=tag)
    #             # TODO make to download at profile date, times ± 2h (within day)
    #             # area and tag supplied externally
    #             if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
    #                 load_source = 'disk'
    #             else:
    #                 raise FileNotFoundError(f'Download attempted but complete ERA5 data not found on disk for {datetime_str}')
    #         else:
    #             raise FileNotFoundError(f'ERA5 data not found on disk for {datetime_str} and local download not attempted')
    # else:
    #     time_label = 'time'
    #     level_label = 'level'
    #     load_source = 'ceda'

    # generating filepaths for dataset at this time/date
    source_fp_gen = util.any_era_fp(datetime_str, 'cams', tag, source=load_source)

    source_fp = source_fp_gen('co2')

    # # getting token for CEDA access
    # if load_source == 'ceda':
    #     token, expires = ceda_io.get_token()
    # else:
    #     token = ''
    profile_ds = data.dataset_load_and_process(source_fp, source=load_source,
                                               token=token, filter=filt_params,
                                               profile_datetime=datetime_str)
    # # load lnsp dataset
    # lnsp_ds = data.dataset_load_and_process(source_fp_ml('lnsp'), source=load_source,
    #                                        token=token, filter=filt_params,
    #                                        profile_datetime=datetime_str)
    
    # # reading in time, lat, lon
    # time_data = lnsp_ds[time_label].values
    # lat_data = lnsp_ds.latitude.values
    # lon_data = lnsp_ds.longitude.values

    # # nesting of data in a list required to make xarray happy
    # if lat_data.ndim == 0:
    #     lat_data = [lat_data]
    #     lon_data = [lon_data]

    # # generate pressure levels from lnsp value(s)
    # p_hl_data = 1.e-2 * half_lvls_ecmwf(lnsp_ds.lnsp.values)
    # p_fl_data = p_lvls_ecmwf(lnsp_ds.lnsp.values)

    # lnsp_ds.close()

    # unit conversions
    for variable in ['ch4', 'co', 'co2']:
        profile_ds[variable] = 1.e3 * profile_ds[variable]  # kg/kg to g/kg

    # change longitude to 'natural' i.e. (-180,180)
    profile_ds['longitude'] = np.mod(profile_ds['longitude'] + 180, 360)-180

    # add in units and other attributes to all variables
    attrs_dict = {
        'ch4':{'units':'g kg-1', 'long_name':'methane'},
        'co':{'units':'g kg-1', 'long_name':'carbon monoxide'},
        'co2':{'units':'g kg-1', 'long_name':'carbon dioxide'},
        'latitude': {'units':'degrees_north', 'long_name':'latitude',},
        'longitude':{'units':'degrees_east', 'long_name':'longitude',},
        'model_level':    {'long_name':'model_level_number',},
        'valid_time':     {'long_name':'forecast_validtime',},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    # profile_ds = xr.load_dataset(source_fp)
    print('CAMS profile loaded successfully')

    return profile_ds


def merge_ghg_to_profile(profile_ds, ghg_ds, dest_dim='level'):
    """
    Just copies one profile (i.e. single lat,lon) of CH4 CO CO2 to an existing dataset
    """
    # levels = profile_ds['level'].values
    # if np.all(np.diff(levels) > 0):
    #     ascending = True
    # elif np.all(np.diff(levels) < 0):
    #     ascending = False

    # sort by pressures instead of levels. sometimes level just blindly tracks the order of values in ERA5 profile
    pressures = profile_ds['p'].values
    if np.all(np.diff(pressures) > 0):  # pressures ascending, so sorted from model level 1 to 137
        ascending = True
    elif np.all(np.diff(pressures) < 0):  # pressures descending, so array sorted from model level 137 to 1
        ascending = False

    ghg_ds = ghg_ds.sortby(['model_level'], ascending=ascending)

    for variable in ['ch4', 'co', 'co2']:
        profile_ds = profile_ds.assign({variable: ([dest_dim,], ghg_ds[variable].values)})
    attrs_dict = {
        'ch4':{'units':'g kg-1', 'long_name':'methane'},
        'co':{'units':'g kg-1', 'long_name':'carbon monoxide'},
        'co2':{'units':'g kg-1', 'long_name':'carbon dioxide'},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)
    return profile_ds



def profile_from_era5_ensemble_spread(datetime_str, point=None, loc_range=None, attempt_download=False, tag=None, dl_area=None):
    
    # checking for specification of location selection
    if point and loc_range:
        raise Exception('Cannot specify both a spatial range and a single point')
    elif point:
        filt_params = point
    elif loc_range:
        filt_params = loc_range

    time_label = 'valid_time'
    level_label = 'model_level'
    # if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
    load_source = 'disk'
        # if attempt_download:
        #     dt_obj = dt.datetime.fromisoformat(datetime_str)
        #     dl_date = dt_obj.strftime('%Y-%m-%d')
        #     hr_pad = 2
        #     dl_times = np.arange(dt_obj.hour-hr_pad, dt_obj.hour+hr_pad+1)
        #     dl_times = dl_times[util.in_range(dl_times, (0,23), incl=True)]
            
        #     print(f'Attempting download for {dl_date}: {len(dl_times)} times selected')
        #     data.download_era5_ml(['lnsp','t','q','o3','z','skt','cc'],
        #                             dl_date, dl_times, 'all', 'enda', 'es',
        #                             area_NWSE=dl_area, tag=tag)
        #     # TODO make to download at profile date, times ± 2h (within day)
        #     # area and tag supplied externally
        #     if data.era5_local_available(datetime_str, tag=tag, filt_params=filt_params):
        #         load_source = 'disk'
        #     else:
        #         raise FileNotFoundError(f'Download attempted but complete ERA5 data not found on disk for {datetime_str}')
        # else:
        #     raise FileNotFoundError(f'ERA5 data not found on disk for {datetime_str} and local download not attempted')

    # generating filepaths for dataset at this time/date
    source_fp_es = util.any_era_fp(datetime_str, 'es', tag, source=load_source)
    source_fp_ml = util.any_era_fp(datetime_str, 'an_ml', tag, source=load_source)
    
    token = ''

    # load lnsp dataset
    lnsp_ds = data.dataset_load_and_process(source_fp_ml('lnsp'), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)
    
    # reading in time, lat, lon
    time_data = lnsp_ds[time_label].values
    lat_data = lnsp_ds.latitude.values
    lon_data = lnsp_ds.longitude.values

    # nesting of data in a list required to make xarray happy
    if lat_data.ndim == 0:
        lat_data = [lat_data]
        lon_data = [lon_data]

    # generate pressure levels from lnsp value(s)
    p_hl_data = 1.e-2 * half_lvls_ecmwf(lnsp_ds.lnsp.values)
    p_fl_data = p_lvls_ecmwf(lnsp_ds.lnsp.values)

    lnsp_ds.close()
    
    # get size of dataset and broadcast pressure levels into that shape
    n_lats_gp = len(lat_data)
    n_lons_gp = len(lon_data)
    n_lvls = 137
    p_fl_data = np.broadcast_to(p_fl_data, (n_lats_gp, n_lons_gp, n_lvls))
    p_hl_data = np.broadcast_to(p_hl_data, (n_lats_gp, n_lons_gp, n_lvls+1))

    # initiate local dataset, using model level as coordinate and pressure as variable
    profile_ds = xr.Dataset(
        data_vars={
        'p':(['latitude','longitude','level'],p_fl_data),
        'p_half':(['latitude','longitude','half_level'],p_hl_data),
        'time':(time_data),
        },
        coords={
        'latitude':(lat_data),
        'longitude':(lon_data),    
        }
    )

    # load in T, q, O3, and geopotential
    for var_ml in ['t', 'q', 'o3']:
        ds = data.dataset_load_and_process(source_fp_es(var_ml), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)

        var_data = ds[var_ml].values[0]

        if point:
            # reallocates data to have correct structure even if just 1 gridpoint chosen
            var_data = var_data[:, np.newaxis, np.newaxis]

        # assign values of variable to dataset, and reorder dimensions
        profile_ds = profile_ds.assign({var_ml:(['level','latitude','longitude'],var_data)})
        profile_ds[var_ml] = profile_ds[var_ml].transpose('latitude','longitude','level')

        ds.close()

    # dealing with skin temp separately as it is defined for single-level, not model level
    skt_ds = data.dataset_load_and_process(source_fp_es('skt'), source=load_source,
                                           token=token, filter=filt_params,
                                           profile_datetime=datetime_str)
    
    skt_data = np.broadcast_to(skt_ds['skt'].values[0], (n_lats_gp, n_lons_gp,))
    profile_ds = profile_ds.assign({'skt':(['latitude','longitude',],skt_data)})

    skt_ds.close()

    # change longitude to 'natural' i.e. (-180,180)
    if load_source == 'ceda':
        profile_ds['longitude'] = np.mod(profile_ds['longitude'] + 180, 360)-180
        profile_ds = profile_ds.sortby('longitude')

    # add in units and other attributes to all variables
    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        'p_half':{'units':'hPa', 'long_name':'half_level_pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'q':    {'units':'kg kg**-1', 'long_name':'specific humidity', 'standard_name':'specific_humidity'},
        'o3':   {'units':'kg kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'skt':  {'units':'K', 'long_name':'skin temperature',},
        'latitude': {'units':'degrees_north', 'long_name':'latitude',},
        'longitude':{'units':'degrees_east', 'long_name':'longitude',},
        'level':    {'long_name':'model_level_number',},
        'time':     {'long_name':'time',},
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    print('ERA5 profile(s) loaded successfully')

    return profile_ds


def total_column_water_vapour(wv_mmr, p_input, p_level_type='half_levels', input_units='g/kg', output_units='kg/m2'):
    # ERA5 in kg/m2, PREFIRE in mm
    # CODE HERE TO DEAL WITH PRESSURE IN ASCENDING/DESCENDING ORDER
    if np.all( np.diff(p_input) > 0 ):
        neg_factor = 1
    elif np.all( np.diff(p_input) < 0 ):
        neg_factor = -1
    else:
        raise ValueError('Input pressure levels not sorted correctly')

    if input_units == 'g/kg':
        unit_factor = 1.e-3
    elif input_units == 'kg/kg':
        unit_factor = 1.

    if p_level_type == 'half_levels':
        
        layer_p_diffs = np.diff(p_input)
        wv_mmr_layers = wv_mmr

    elif p_level_type == 'full_levels':

        layer_p_diffs = np.diff(p_input)
        wv_mmr_layers = np.convolve(wv_mmr, [0.5, 0.5], mode='valid')

    tcwv = neg_factor * np.sum(unit_factor*wv_mmr_layers * 1.e2*layer_p_diffs) / g_0

    if output_units == 'mm':
        tcwv = 1.e3 * tcwv / rho_water

    return tcwv



def era_profile_following_sonde(sonde_fp, load_local=True, load_ghg=True):

    sonde_data = data.load_radiosonde_nc(sonde_fp)

    # This has been added in an attempt to fix an issue where too many or too few layers are loaded (i.e. 136 or 138) but does not work.
    alt_pad = 0.005 # km

    times = list(set( [util.round_to_hours(dt.datetime.fromisoformat(str(timeval))).strftime('%Y-%m-%dT%H:%M') for timeval in sonde_data.time.values] ))

    datasets = []
    for datetime_str in times:
        era5_hr_data = profile_from_ml_era5_new(datetime_str, loc_range=[-75,-69,43,46], tag='eastCanada', load_local=load_local).squeeze()

        era5_hr_data = era5_hr_data.sortby(['level','half_level'], ascending=False)
        era5_hr_data = add_altitudes_era(era5_hr_data)
        datasets += [era5_hr_data]

    era5_data = xr.concat(datasets, 'time')


    sonde_lat_era5_res = util.round_to_res(sonde_data['LATITUDE'].values)
    sonde_lon_era5_res = util.round_to_res(sonde_data['LONGITUDE'].values)
    sonde_times_nearest_hr = [np.datetime64(util.round_to_hours(dt.datetime.fromisoformat(str(timeval)))) for timeval in sonde_data.time.values]

    lat_crossings = np.nonzero(np.diff(sonde_lat_era5_res))[0] + 1
    lon_crossings = np.nonzero(np.diff(sonde_lon_era5_res))[0] + 1
    time_crossings = np.nonzero(np.diff(sonde_times_nearest_hr))[0] + 1

    last_idx = len(sonde_lat_era5_res)-1

    all_crossings = np.sort(list(set( np.concatenate( ([0], lat_crossings, lon_crossings, time_crossings, [last_idx]) ) )))


    hybrid_data = {}

    vars_todo = ['p', 't', 'altitude', 'q', 'o3']

    # bottom of profile

    start_lat = sonde_lat_era5_res[0]
    start_lon = sonde_lon_era5_res[0]
    start_time = sonde_times_nearest_hr[0]

    start_profile = era5_data.sel(latitude=start_lat, longitude=start_lon, time=start_time)

    start_alt = sonde_data['ALTITUDE'].values[0] / 1.e3

    first_gc_altitude_mask = util.in_range(start_profile['altitude'].values, (-2., start_alt-alt_pad), incl=True)
    for variable in vars_todo:

        gridcell_data = start_profile[variable][first_gc_altitude_mask].values
        hybrid_data[variable] = gridcell_data
        
    hybrid_data['skt'] = start_profile['skt'].values
    hybrid_data['z'] = start_profile['z'].values


    # middle of profile

    for i in range(len(all_crossings)-1):

        curr_idx = all_crossings[i]
        next_idx = all_crossings[i+1]

        curr_lat = sonde_lat_era5_res[curr_idx]
        curr_lon = sonde_lon_era5_res[curr_idx]
        curr_time = sonde_times_nearest_hr[curr_idx]

        current_profile = era5_data.sel(latitude=curr_lat, longitude=curr_lon, time=curr_time)

        curr_alt = sonde_data['ALTITUDE'].values[curr_idx] / 1.e3
        next_alt = sonde_data['ALTITUDE'].values[next_idx] / 1.e3

        gridcell_altitude_mask = util.in_range(current_profile['altitude'].values, (curr_alt, next_alt), incl=True)

        for variable in vars_todo:

            gridcell_data = current_profile[variable][gridcell_altitude_mask].values
            hybrid_data[variable] = np.append(hybrid_data[variable], gridcell_data)


    # top of profile

    final_lat = sonde_lat_era5_res[last_idx]
    final_lon = sonde_lon_era5_res[last_idx]
    final_time = sonde_times_nearest_hr[last_idx]

    final_profile = era5_data.sel(latitude=final_lat, longitude=final_lon, time=final_time)

    final_alt = sonde_data['ALTITUDE'].values[last_idx] / 1.e3

    last_gc_altitude_mask = util.in_range(final_profile['altitude'].values, (final_alt+alt_pad, 150.), incl=True)
    for variable in vars_todo:

        gridcell_data = final_profile[variable][last_gc_altitude_mask].values
        hybrid_data[variable] = np.append(hybrid_data[variable], gridcell_data)
    
    
    profile_following = {key : (['p'], values) for key, values in hybrid_data.items()}
    for variable in ['p', 'z', 'skt']:
        profile_following[variable] = hybrid_data[variable]

    # for subsequent easy code use
    profile_following['sonde_bounds_press'] = [sonde_data['PRESSURE_INSITU'].values.max(), sonde_data['PRESSURE_INSITU'].values.min()]
    profile_following['era5_latitude'] = start_lat
    profile_following['era5_longitude'] = start_lon
    profile_following['era5_time'] = start_time
    profile_following['level'] = (['p'], np.arange(era5_data.level.size, 0, -1))

    profile_ds = xr.Dataset(profile_following)
    if load_ghg:
        cams_ghg_ds = ghg_profile_cams(str(sonde_data.time.values[0]), point=(sonde_data['LATITUDE'].values[0], sonde_data['LONGITUDE'].values[0]) , tag='eastCanada')
        profile_ds = merge_ghg_to_profile(profile_ds, cams_ghg_ds, dest_dim='p')

    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        'p_half':{'units':'hPa', 'long_name':'half_level_pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'q':    {'units':'g kg**-1', 'long_name':'water vapour mass mixing ratio', 'standard_name':'humidity_mixing_ratio'},
        'o3':   {'units':'g kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'z':    {'units':'km', 'long_name':'surface altitude', 'standard_name':'surface_altitude'},
        'altitude': {'units':'km', 'long_name':'altitude'},
        'skt':  {'units':'K', 'long_name':'skin temperature',},
    }

    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    profile_ds = profile_ds.assign_attrs({'Info': 'Profile created from ERA5 data following the 3D radiosonde path. Bottom and top of profile filled in at initial and final radiosonde location/time respectively'})

    return profile_ds



def sonde_era_station_hybrid(filepath_sonde, filepath_met, seconds_to_avg=150, sfc_altitude=0.130,
                             full_profile_start_alt=0.149, heights_to_keep=[1., 2., 4.35, 7.1, 10.]):

    sentinel_lat, sentinel_lon = 45.535021, -73.149006

    # get sonde-ERA5 hybrid for most of profile already sorted
    sonde_hybrid_profile = sonde_era_hybrid(filepath_sonde, 200, 'custom_brb', 'p')
    lowest_sonde_level = np.min(np.nonzero(['sonde' in level_source for level_source in sonde_hybrid_profile['data_source'].values]))
    launch_time = str(sonde_hybrid_profile['sonde_time'].values[lowest_sonde_level])
    launch_coords = (sonde_hybrid_profile['sonde_latitude'].values[lowest_sonde_level],
                     sonde_hybrid_profile['sonde_longitude'].values[lowest_sonde_level])
    
    higher_mask = np.nonzero(sonde_hybrid_profile['z'].values >= full_profile_start_alt)[0]
    sonde_era_above = sonde_hybrid_profile.isel(p=higher_mask)

    # loading processed met dataset, time-averaging across subset
    local_met_ds = xr.load_dataset(filepath_met)
    met_ds_subset = data.dataset_time_subset(local_met_ds, launch_time, seconds_to_avg)
    met_ds_subset_avg = met_ds_subset.mean('time')

    # selecting retained heights, converting temperature and humidity values
    req_heights_avgs_raw = met_ds_subset_avg.sel(height_agl=heights_to_keep)
    req_heights_avgs_raw = req_heights_avgs_raw.sortby(['height_agl'], ascending=True)
    n_sentinel = req_heights_avgs_raw['height_agl'].size

    sentinel_temps = req_heights_avgs_raw['temperature'] + 273.15
    sentinel_wv_mmrs = 1.e3 * convert_rh_to_mass_mixing_ratio(req_heights_avgs_raw['rel_hum'], req_heights_avgs_raw['pressure'].values, sentinel_temps)

    sentinel_mean_time = met_ds_subset.time.mean().values
    sentinel_rdgs = met_ds_subset.time.size

    sonde_data = data.load_radiosonde_nc(filepath_sonde)

    # using radiosonde p(z) scale near-surface (linear fit) to calculate p(z) for sentinel measurements
    early_slice = slice(0,6)
    low_sonde_pressures = sonde_data['PRESSURE_INSITU'].values[early_slice]
    low_sonde_altitudes = 1.e-3 * sonde_data['ALTITUDE'].values[early_slice]

    p_with_z_sonde = np.polyfit(low_sonde_altitudes, low_sonde_pressures, 1)

    sentinel_alts = 1.e-3 * req_heights_avgs_raw['height_agl'].values + sfc_altitude
    sentinel_pressures = np.polyval(p_with_z_sonde, sentinel_alts)


    #     # time_avg = np.average(sonde_time[mask_single_level].astype(int)).astype(np.dtypes.DateTime64DType)
    #     time_avg = pd.to_datetime( np.average(sonde_time[mask_single_level].astype(np.int64)), unit='us')
    #     lat_avg = np.average(sonde_lat[mask_single_level])
    #     lon_avg = np.average(sonde_lon[mask_single_level])

    #     p[i], t[i], z[i], wv_mmr[i], n_readings[i] = p_fl, t_fl, z_fl, wv_mmr_fl, n_fl
    #     sonde_times_avg[i], sonde_lats_avg[i], sonde_lons_avg[i] = time_avg, lat_avg, lon_avg

    # sonde_times_avg = [str(time) for time in sonde_times_avg]

    # sonde_times_avg = sonde_times_avg.astype('<M8[ns]')

    # loading in relevant ERA5 for launch time and location
    # would need to rewrite here to have ERA5 following sonde, including check that sonde within data bounding box

    era5_ds = profile_from_ml_era5_new(launch_time, point=launch_coords, attempt_download=False,
                                               tag='eastCanada', dl_area=[48, -77, 43, -70]).squeeze()
    era5_ds = era5_ds.sortby(['p','p_half'], ascending=False)
    era5_ds = add_altitudes_era(era5_ds)

    # load in CAMS data and add to existing profile dataset
    cams_ghg_ds = ghg_profile_cams(launch_time, point=launch_coords, tag='eastCanada')
    era5_ds = merge_ghg_to_profile(era5_ds, cams_ghg_ds)

    # HERE we just need to have ERA5 fields not already loaded interpolated to sentinel levels:
    # o3, co2, co, ch4
    o3_interp_era, co2_interp, co_interp, ch4_interp = data.nondim_interp(era5_ds, sentinel_pressures, 'p', ['o3', 'co2', 'co', 'ch4'])

    
    # NEED TO SORT SENTINEL THINGS CORRECTLY AND THEN ADD TO CONCATENATION

    # SECOND: take ERA5 top and bottom levels as they come
    # could write this as a 'nest arrays' function
    p_all_sources = np.concatenate([sentinel_pressures, sonde_era_above['p'].values])
    t_all_sources = np.concatenate([sentinel_temps, sonde_era_above['t'].values])
    z_all_sources = np.concatenate([sentinel_alts, sonde_era_above['z'].values])
    wv_mmr_all_sources = np.concatenate([sentinel_wv_mmrs, sonde_era_above['q'].values])

    sentinel_avg_times = np.full(n_sentinel, sentinel_mean_time, dtype='<M8[us]')
    sentinel_avg_times = [str(time) for time in sentinel_avg_times]
    sentinel_lats = np.full(n_sentinel, sentinel_lat)
    sentinel_lons = np.full(n_sentinel, sentinel_lon)
    sentinel_n_rdgs = np.full(n_sentinel, sentinel_rdgs)

    meas_times_avg = np.concatenate([sentinel_avg_times,
                                     [str(time) for time in sonde_era_above['sonde_time'].values]])
    meas_times_avg = np.asarray(meas_times_avg, dtype=np.datetime64)
    meas_lats_avg = np.concatenate([sentinel_lats, sonde_era_above['sonde_latitude'].values])
    meas_lons_avg = np.concatenate([sentinel_lons, sonde_era_above['sonde_longitude'].values])
    meas_n_rdgs = np.concatenate([sentinel_n_rdgs, sonde_era_above['sonde_readings'].values])
    data_source = np.concatenate([np.full(n_sentinel, 'Sentinel HMP155A'), sonde_era_above['data_source'].values])

    # THIRD: add in skt, and o3 interpolated to vertical levels
    skintemp_era = era5_ds.skt.values
    o3_interp_era, co2_interp, co_interp, ch4_interp = data.nondim_interp(era5_ds, p_all_sources, 'p', ['o3', 'co2', 'co', 'ch4'])

    profile_ds = sonde_era_above.copy().drop_dims('p')

    # PROFILE: things as coord of pressure
    profile_ds = profile_ds.assign({
        'p':(p_all_sources),
        't': (['p'], t_all_sources),
        'z': (['p'], z_all_sources),
        'q': (['p'], wv_mmr_all_sources),
        'o3': (['p'], o3_interp_era),
        'co2': (['p'], co2_interp),
        'co': (['p'], co_interp),
        'ch4': (['p'], ch4_interp),
        'skt': (skintemp_era),
        'data_source': (['p'], data_source),
        'measurement_time': (['p'], meas_times_avg),
        'measurement_latitude': (['p'], meas_lats_avg),
        'measurement_longitude': (['p'], meas_lons_avg),
        'measurement_readings': (['p'], meas_n_rdgs),
        }
    )

    attrs_dict = {
        'p':    {'units':'hPa', 'long_name':'pressure',},
        't':    {'units':'K', 'long_name':'temperature', 'standard_name':'air_temperature'},
        'z':    {'units':'km', 'long_name':'geometric altitude', 'standard_name':'altitude'},
        'q':    {'units':'g kg**-1', 'long_name':'water vapour mass mixing ratio', 'standard_name':'humidity_mixing_ratio'},
        'o3':   {'units':'g kg**-1', 'long_name':'ozone mass mixing ratio', 'standard_name':'mass_fraction_of_ozone_in_air'},
        'co2':   {'units':'g kg**-1', 'long_name':'carbon dioxide mass mixing ratio'},
        'co':   {'units':'g kg**-1', 'long_name':'carbon monoxide mass mixing ratio'},
        'ch4':   {'units':'g kg**-1', 'long_name':'methane mass mixing ratio'},
        'measurement_time':     {'long_name':'in-situ measurement averaged time',},
        'measurement_latitude': {'units':'degrees_north', 'long_name':'in-situ measurement (averaged) latitude',},
        'measurement_longitude':{'units':'degrees_east', 'long_name':'in-situ measurement (averaged) longitude',},
        'measurement_readings': {'long_name':'number of in-situ readings averaged',},
        'era5_time':     {'long_name':'ERA5 profile time',},
        'data_source':{'description':'source of data for p, t, z, q at the given vertical level'}
    }
    profile_ds = data.add_attrs(profile_ds, attrs_dict)

    info_dict = {'name':'hybrid radiosonde-ERA5-Sentinel profile',
                 'sentinel_processing': f'readings taken averaged over a time window of {2*seconds_to_avg} s, with some levels discarded',
                 'era_processing':'skt taken directly, o3 and GHGs interpolated onto pressure layers. all fields in layers outside of radiosonde&sentinel range defined by ERA5'}

    profile_ds = profile_ds.assign_attrs(info_dict)

    return profile_ds



def hybrid_test_profile(full_profile, met_ds_subset_avg, sonde_data, full_profile_type, sfc_altitude, full_profile_start_height, use_10m):


    if full_profile_type == 'era':
        higher_mask = np.nonzero(full_profile['p'].values < full_profile_start_height)[0]
        alt_label='altitude'

    elif full_profile_type == 'sonde':
        higher_mask = np.nonzero(full_profile['z'].values > full_profile_start_height)[0]
        alt_label='z'

    higher_mask = np.asarray(higher_mask)

    heights_to_keep = [1., 2., 4.35, 7.1]
    if use_10m:
        heights_to_keep += [10.]

    req_heights_avgs_raw = met_ds_subset_avg.sel(height_agl=heights_to_keep)

    sentinel_temps = req_heights_avgs_raw['temperature'].values + 273.15
    sentinel_wv_mmrs = 1.e3 * convert_rh_to_mass_mixing_ratio(req_heights_avgs_raw['rel_hum'].values, req_heights_avgs_raw['pressure'].values, sentinel_temps)

    # using radiosonde p(z) scale near-surface (linear fit) to calculate p(z) for sentinel measurements
    early_slice = slice(0,6)
    low_sonde_pressures = sonde_data['PRESSURE_INSITU'].values[early_slice]
    low_sonde_altitudes = 1.e-3 * sonde_data['ALTITUDE'].values[early_slice]

    p_with_z_sonde = np.polyfit(low_sonde_altitudes, low_sonde_pressures, 1)

    sentinel_alts = 1.e-3 * req_heights_avgs_raw['height_agl'].values + sfc_altitude
    sentinel_pressures = np.polyval(p_with_z_sonde, sentinel_alts)

    label = f'{full_profile_type}_start{full_profile_start_height:.3f}'
    if use_10m:
        label = label+'_use10m'

    t = np.concatenate([sentinel_temps, full_profile['t'].values[higher_mask]])
    q = np.concatenate([sentinel_wv_mmrs, full_profile['q'].values[higher_mask]])
    z = np.concatenate([sentinel_alts, full_profile[alt_label][higher_mask].values])
    p = np.concatenate([sentinel_pressures, full_profile['p'][higher_mask].values])

    o3 = np.interp(p, full_profile['p'].values, full_profile['o3'].values )

    return t, q, z, p, o3, label


def scale_sonde_h2o_in_hybrid(profile, scale_factor):
    source_search = 'sonde'
    var_to_scale = 'q'

    if scale_factor != 1.:

        selected_level_mask = np.array([source_search in data_source.casefold() for data_source in profile['data_source'].values])
        var_scaling_masked = (1. - scale_factor) * selected_level_mask + 1
        wat_vap_attrs = profile[var_to_scale].attrs
        profile[var_to_scale] = var_scaling_masked * profile[var_to_scale]
        profile[var_to_scale] = profile[var_to_scale].assign_attrs(wat_vap_attrs)

        print(f'radiosonde {var_to_scale} levels scaled by {100 * (scale_factor - 1.):.1f}%')

    else:
        pass

    return profile
