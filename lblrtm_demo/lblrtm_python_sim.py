"""
B Pery, October 2026
Example code for running LBLRTM and simulating FINESSE with Python

Before running, ensure that LBLRTM is set up correctly on your system, and that you have
set the _lblpath variable in the bp_core/lblrtm.py file.
"""
import bp_core as bjp
import xarray as xr
import numpy as np

# %% this tag allows creation of code 'cells' for interactive Python sessions
# specifying inputs to programme
profile_fp = './example_profile.nc'
output_location = '/net/thunder/users/bjp224/demo/LBLRTM_out/'

finesse_wn_range = np.array([360, 1600]) 
finesse_wn_range_padded = finesse_wn_range + [-20, 20] # add a padding to remove edge effects
lbl_atm = 5  # (subarctic winter)
profile_txt = 'Example Arctic winter profile'

obs_pressure = 1000. # pressure (hPa) at observation site
view_angle = 0. # FINESSE looking straight up 
boundary_pressure = 0.001 # start simulating atmosphere down from 0.001 hPa
t_boundary = 2.7  # K surface temp of outer space

# %%
# Loading in atmospheric profile and making profile dictionaries
profile_ds = xr.load_dataset(profile_fp)

profile_input, gas_input, _ = bjp.lblrtm.make_profile_dicts(profile_ds,
                                        change_input_keys={'pressure':'pressure_level', 'altitude':'z'})

# %%
"""
Running LBLRTM: see funciton bjp.lblrtm.slant_path_run() for an example of how this code
can be functionalised flexibly (but don't copy this, you don't need all those switches!)
"""

# getting path of LBLRTM executable
lbl_exe_loc = bjp.lblrtm._lblpath

# writing TAPE5 (LBLRTM input) file for this case
bjp.lblrtm.write_lblrtm_tape5(
    tape5_fp=lbl_exe_loc+'TAPE5', # not strictly needed but good to check you're thinking about it!
    config_fp=lbl_exe_loc+'lblrtm_config', # likewise

    profile_dict=profile_input,
    gas_input_dict=gas_input,
    profile_description=profile_txt,
    xsect_opt=False, # switch off cross-section gases
    atm_gas_defaults=lbl_atm,

    wn_range=finesse_wn_range_padded, # adding a padding at edge of band
    boundary_temp=t_boundary,
    obs_alt=obs_pressure,
    end_or_tan_alt=boundary_pressure,
    zen_angle=view_angle,
    surf_emis_coeffs=[1.],
    surf_refl_coeffs=[0.],
)

# Writing a quick description to go in to lblrtm_info.txt file
output_descriptor = f'Example Arctic winter profile on {profile_ds['valid_time'].values}\n' + \
                    f'Location: {profile_ds['latitude'].values}°N, {profile_ds['longitude'].values}°E\n' + \
                    f'Source: ERA5 on pressure levels (37-level)'

# Running LBLRTM executable, which will move files to output_location
bjp.lblrtm.run_LBLRTM(
    output_path=output_location,
    file_write=output_descriptor,
)

# %%
"""
Simulating FINESSE
"""
# read in binary format LBLRTM output file
lbl_spec = bjp.lblrtm.read_tape12(output_location + 'TAPE12')  # LBLRTM output file read in

# pre-processing of spectrum: this step modifies the input hi-res spectrum to match an ideal
# Fourier-transform spectrometer with the FINESSE optical path difference
wn_FINESSE, radiance_FINESSE = bjp.inst.process_spectrum_general(lbl_spec.wn.data, lbl_spec.radiance.data,
                                                                 0.01, *finesse_wn_range_padded, 1.21)

# apply the FINESSE spectrally-varying line shape (which is stored in bins)
lbl_FINESSE_spec = bjp.inst.apply_ils_bins(radiance_FINESSE, wn_FINESSE,
                                           inst_res=0.2, 
                                           spec_range=finesse_wn_range+[0, 0.2] # adding the extra helps match FINESSE obs
                                           )

# save this spectrum to the output location
lbl_FINESSE_spec.to_netcdf(output_location + 'spectrum_FINESSE.nc')
print('FINESSE spectrum simulated and saved')

# %%
# test plotting

lbl_FINESSE_spec['radiance'].plot()

# %%
