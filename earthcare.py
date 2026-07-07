import numpy as np
import xarray as xr
import pandas as pd
from . import met, planck, util

msi_translation = {
    # translates from standard channel names to EarthCARE MSI channel keys
    'VIS0.6'    :'VIS',
    'VNIR0.8'   :'VNIR',
    'NIR1.6'    :'SWIR1',
    'NIR2.25'   :'SWIR2',
    'IR8.7'     :'TIR1',
    'IR10.8'    :'TIR2',
    'IR12.0'    :'TIR3',
    
    'channel_name_dim':'band',
}

DEFAULT_NADIR_PIX = 278  # cross-track index of the nadir-view pixel

def load_ec_ds(fn, trim=False):

    if 'earthcare_ifs' in fn:
        groupname = ''
    else:
        groupname = 'ScienceData'

    try:
        ds = xr.load_dataset(fn, group=groupname, engine='h5netcdf')
    except ValueError:
        ds = xr.load_dataset(fn, group=groupname, engine='h5netcdf', decode_times=False)
        ds = xr.decode_cf( ds.sel( along_track=(ds['time']<1.e36 ) ) )

    if trim:
        ds = trim_ec_frame_latitudes(ds, fn)

    return ds


frame_extent_lookup = dict(
    # ID = (start, end)
    A = (-22.5, 22.5),
    B = (22.5, 67.5),
    C = (67.5, 90.),
    D = (67.5, 22.5),
    E = (22.5, -22.5),
    F = (-22.5, -67.5),
    G = (-67.5, -90.),
    H = (-67.5, -22.5),
)


def trim_ec_frame_latitudes(ds, fn):

    frame_id = get_ec_frame_id(fn)

    frame_extents = frame_extent_lookup[frame_id]

    latitudes = np.sort(frame_extents) 

    if len(ds['latitude'].dims) == 2:
        lat_preselection = [slice(None), DEFAULT_NADIR_PIX]

    elif len(ds['latitude'].dims) == 1:
        lat_preselection = [slice(None)]

    ds = ds.sel(along_track = util.in_range(ds['latitude'][*lat_preselection], latitudes))

    return ds


def get_ec_frame_id(fn):

    if 'earthcare_ifs' in fn:
        orbit_and_frame_id = xr.load_dataset(fn, engine='h5netcdf').frame_id
        frame_id = orbit_and_frame_id[-1]  # last digit of string

    else:
        groupname = '/HeaderData/VariableProductHeader/MainProductHeader'
        frame_id = str(xr.load_dataset(fn, group=groupname, engine='h5netcdf')['frameID'].data)

    return frame_id


def mask_msi_edge_pixels(msi_ds):
    # trims edges of across-track domain
    # probably a more rigorous way to do it but fine for now

    msi_ds_masked = msi_ds.copy(deep=True)
    msi_ds_masked['across_track'] = msi_ds['across_track'].values
    
    n_across_track = msi_ds_masked['across_track'].size
    n_edge_mask = 24

    msi_ds_masked = msi_ds_masked.sel(
        across_track=slice(n_edge_mask, n_across_track-n_edge_mask-1))

    return msi_ds_masked


def get_msi_reflectances(msi_ds):

    irradiance_recast = msi_ds['solar_spectral_irradiance'].rename({'VNS_band':'band'})
    reflectances = np.pi * msi_ds['pixel_values'].sel(band=['VIS', 'VNIR', 'SWIR1', 'SWIR2']) / (irradiance_recast * np.sin(msi_ds['solar_elevation_angle'] * np.pi/180))

    return reflectances


def get_msi_brightness_temps(msi_ds):

    brightness_temps = msi_ds['pixel_values'].sel(band=['TIR1', 'TIR2', 'TIR3'])

    return brightness_temps


def l2_acm_cap_profile_to_lrt_ic_file(single_profile, ic_fn='', capping=''):

    if ic_fn == '':
        ic_fn = 'ic_file.tmp'

    r_eff = 1.e6 * single_profile['ice_effective_radius'].values
    iwc = 1.e3 * single_profile['ice_water_content'].values
    height_km = 1.e-3 * single_profile['height'].values

    if 'baum_v36' in capping:  
        cap_vals = (5, 60)
        r_eff = util.cap_array(r_eff, cap_vals)
    elif 'baum' in capping or 'yang2013' in capping:
        cap_vals = (5, 90)
        r_eff = util.cap_array(r_eff, cap_vals)
    
    cloud_region = np.nonzero(iwc)[0]

    iwc_write = np.append([0], iwc[cloud_region])
    r_eff_write = np.append([0], r_eff[cloud_region])
    height_write = np.append(height_km[np.min(cloud_region)-1], height_km[cloud_region])

    ic_cols_to_write = np.array([height_write, iwc_write, r_eff_write,])
    ic_file_hdr = '#    z[km]  IWC[g/m3]  Reff[um]'
    np.savetxt(ic_fn, ic_cols_to_write.T, fmt=['%11.3f', '%11.5f', '%11.2f'], header=ic_file_hdr)


def ifs_to_lrt_atmfile(profile_ds, atm_fn='', default_profile='afglus'):
    """
    default_profile could be one of ['afglms', 'afglmw', 'afglss', 'afglsw', 'afglt', 'afglus']
    """
    atm_data_location = '/users/bjp224/libRadtran-2.0.6/data/atmmod/'

    if atm_fn == '':
        atm_fn = 'atmfile.tmp'

    altitude = 1.e-3 * profile_ds['height_fl'].values
    pressure = profile_ds['pressure_fl'].values
    temperature = profile_ds['temperature_fl'].values
    spec_hum = profile_ds['q'].values
    o3_mmr = profile_ds['o3_mmr'].values

    # convert H2O spec hum kg kg-1 to cm-3

    h2o_num_density = met.gas_number_density(spec_hum, met.molar_masses['h2o'], pressure, temperature, spec_hum)

    # convert O3 mmr kg kg-1 to cm-3
    o3_num_density = met.gas_number_density(o3_mmr, met.molar_masses['o3'], pressure, temperature, spec_hum)

    # make array (9 entries) for writing to txt
    
    total_num_density = 1.e-6 * np.divide( pressure, planck._k_B * temperature)

    # TODO TO CALCULATE THESE: read in US standard atmosphere to get profile variation as fn of pressure
    # get USATM molecule ratio (like ppmv) by dividing by USATM total air density cm-3.
    default_mols = ['o2(cm-3)', 'co2(cm-3)', 'no2(cm-3)']
    def_atm_df = load_lrt_atmfile(atm_data_location + default_profile + '.dat')
    def_vmrs = def_atm_df[default_mols].div(def_atm_df['air(cm-3)'], axis=0)
    def_vmrs['z(km)'] = def_atm_df['z(km)']
    def_vmrs['p(mb)'] = def_atm_df['p(mb)']
    
    prof_df = pd.DataFrame(dict(
        z=altitude,
        p=pressure,
    ))
    for mol in default_mols:
        prof_df[mol.replace('(cm-3)', '(vmr)')] = np.interp(prof_df['p'], def_atm_df['p(mb)'], def_vmrs[mol],)
    
    default_mol_num_densities = prof_df[[mol.replace('(cm-3)', '(vmr)') for mol in default_mols]].multiply(total_num_density, axis='index')
    default_mol_num_densities['p'] = prof_df['p']
    default_mol_num_densities['z'] = prof_df['z']
    default_mol_num_densities.rename(
        {mol.replace('(cm-3)', '(vmr)'):mol for mol in default_mols}, axis=1, inplace=True)
    
    # for CO2, should also set mixing_ratio co2 current_value in the input run file.
    o2_num_density = default_mol_num_densities['o2(cm-3)']
    co2_num_density = default_mol_num_densities['co2(cm-3)']
    no2_num_density = default_mol_num_densities['no2(cm-3)']

    atmosphere_cols_to_write = np.array([altitude, 1e-2*pressure, temperature, total_num_density, o3_num_density, o2_num_density, h2o_num_density, co2_num_density, no2_num_density])
    atmfile_hdr = '#     z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)'
    np.savetxt(atm_fn, atmosphere_cols_to_write.T, fmt=['%12.3f', '%12.5f', '%12.3f'] + ['%12.6E']*6, header=atmfile_hdr)


def load_lrt_atmfile(filepath, n_hdr=2):

    headers = pd.read_csv(filepath, sep=r'\s+', skiprows=n_hdr-1, nrows=0).columns[1:]
    atm_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=n_hdr, names=headers)
    
    return atm_file