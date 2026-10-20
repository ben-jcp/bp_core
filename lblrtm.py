import xarray as xr
import numpy as np
import os
import subprocess as sub
import sys
import time
import re
import datetime as dt

# path to the directory containing AER 'RC_utils.py' file used for reading Fortran binary files
# aer_common_path = '/net/thunder/users/bjp224/common'
# sys.path.append(aer_common_path)
from . import panel_file as pf_rd
from . import util
from .write_tape_5_BP import write_tape5  # this is kept in for access
from .aer_tools import RC_utils

_lblpath = '/net/thunder/users/bjp224/lblrtm_12.17/'


gas_list = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3', 'OH', 'HF', 'HCL', 'HBR',
            'HI', 'CLO', 'OCS', 'H2CO', 'HOCL', 'N2', 'HCN', 'CH3CL', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S',
            'HCOOH', 'HO2', 'O', 'CLONO2', 'NO+', 'HOBR', 'C2H4', 'CH3OH']


gas_number_dict = dict(zip(gas_list, np.arange(1, 40)))


xsect_gas_list = ['HNO4', 'CH3COCH3', 'CF3CH2CF3', 'C2F6', 'CHF2CL', 'C2F4CL2', 'CHClF2', 'C2F3CL3', 'CHCl2C2F5', 'CF2CL2',
                  'SF6', 'SO2', 'C2CL3F3', 'CH3C(O)CH3', 'CFC13', 'CHCL2F', 'F13', 'NF3', 'CFC14', 'BRO', 'HCHO', 'ACETONE',
                  'CFC113', 'NO2', 'CFC21', 'F14', 'N2O5', 'CH3CN', 'CFCL3', 'CCl2FCH3', 'F21', 'CH3CClF2', 'FURAN', 'CHF2CH2CF3',
                  'F113', 'CCLF3', 'F115', 'F11', 'CLNO3', 'CFC115', 'CF3CH3', 'CHCl2CF3', 'CCL3F', 'CH2F2', 'F11CCL2F2',
                  'C2HCl2F3CF2', 'PAN', 'CFC11', 'F114', 'HNO3', 'CFC22', 'CFC114', 'F22', 'CHF3', 'ACETICACI', 'GLYCOLALD',
                  'CLONO2', 'CFC12', 'PROPENE', 'CCL4', 'F12', 'ACET', 'CH3CHF2', 'ISOP', 'C2CL2F4', 'C2CLF5', 'CF4', 'CHCLF2',
                  'CHF2CF3']


alias_dict = {
    'hirac_opt': 'IHIRAC',
    'lblf4_opt': 'ILBLF4',
    'continuum_opt': 'ICNTNM',
    'aerosol_opt': 'IAERSL',
    'calc_result_opt': 'IEMIT',
    'scan_result_opt': 'ISCAN',
    'filter_opt': 'IFILTR',
    'plot_opt': 'IPLOT',
    'test_opt': 'ITEST',
    'lblatm_opt': 'IATM',
    'merge_opt': 'IMRG',
    'optical_depth_opt': 'IOD',
    'xsect_opt': 'IXSECT',

    'h2o_self_cntm_scaling': 'XSELF',
    'h2o_frgn_cntm_scaling': 'XFRGN',
    'co2_cntm_scaling': 'XCO2C',
    'o3_cntm_scaling': 'XO3CN',
    'o2_cntm_scaling': 'XO2CN',
    'n2_cntm_scaling': 'XN2CN',
    'rayleigh_scaling': 'XRAYL',

    'min_wn':'V1',
    'max_wn':'V2',
    'samples_per_mean_hw': 'SAMPLE',
    'dv_setting': 'DVSET',
    'ave_coll_broad_hw': 'ALFAL0',
    'ave_mol_mass_doppler_hw': 'AVMASS',
    'line_rejection_od': 'DPTMIN',
    'cont_od_rejection_factor': 'DPTFAC',
    'line_rejection_recort_opt': 'ILNFLG',
    'dv_monochr_output_grid': 'DVOUT',
    'n_molecules_scaled': 'NMOL_SCAL',

    'boundary_temp': 'TBOUND',
    'surf_emis_coeffs': 'SREMIS(N)',
    'surf_refl_coeffs': 'SRREFL(N)',
    'surf_refl_type': 'surf_refl',

    'jacobian_species_number': 'NSPCRT',

    'od_filepath': 'PTHODL',
    'num_od_layers': 'LAYTOT',

    'internal_model_atm': 'MODEL',
    'path_type_opt': 'ITYPE',
    'zero_small_abs': 'ZERO',
    'short_print_opt': 'NOPRNT',
    'n_species': 'NMOL',
    'layer_dat_write_opt': 'IPUNCH',
    'tape7_control_opt': 'IFXTYP',
    'tape7_gas_unit_opt': 'MUNITS',
    'earth_radius': 'RE',
    'space_altitude': 'HSPACE',
    'refraction_freq': 'VBAR',
    'reference_latitude': 'REF_LAT',

    'obs_alt': 'H1',
    'end_or_tan_alt': 'H2',
    'zen_angle': 'ANGLE',
    'path_length': 'RANGE',
    'path_angle': 'BETA',
    'tan_height_opt': 'LEN',
    'sat_obs_height': 'HOBS',

    'pressure_unit_ctrl': 'JCHARP',
    'temperature_unit_ctrl': 'JCHART',
    'gas_input_format_ctrl': 'JLONG',
}


def read_tape12(filepath, load_transmittances=False):
    """
    Function to read Fortran formatted binary files as output by LBLRTM, including (but not limited to) TAPE12.

    filepath:   path of file to be read
    load_transmittances:    if set to True, the programme will try to load transmittances as well as radiances.

    lblrtm_output:      xr.Dataset containing the information loaded from the file.

    NOTE that this code currently naively assumes units for radiance calculations; the units output in the Dataset
    will be incorrect for other kinds of calculations (e.g. optical depth, Jacobian)
    """


    try:
        tape12_data = pf_rd.panel_file(filepath, do_load_data=True)

        wns = tape12_data.v
        rads = tape12_data.data1
        trans = tape12_data.data2

        aer_loaded = False

    except UnicodeDecodeError:
        print('\'Panel file\' loading failed, attempting with AER tool')

        wns, rads = RC_utils.readBinary(filepath)
        aer_loaded = True

    wavenumbers = xr.DataArray(wns,
                               coords={'wn':wns},
                               attrs={'units':'cm**-1'})
    radiances = xr.DataArray(rads,
                             coords={'wn':wns},
                             attrs={'units':'W /( cm**2 sr cm**-1 )'})

    ds_dict = {'radiance':radiances}

    # can only load transmittances with Panel file
    if load_transmittances and not aer_loaded:

        transmittances = xr.DataArray(trans,
                                    coords={'wn':wavenumbers},
                                    attrs={'units':'1'})
        
        ds_dict['transmittance'] = transmittances
    
    lblrtm_output = xr.Dataset(
        data_vars=ds_dict,
        coords={'wn':wavenumbers} 
    )

    return lblrtm_output


def run_LBLRTM(output_path:str, file_write=False, lblrtm_path=_lblpath, lbl_exe_name='lblrtm_v12.17_linux_gnu_dbl', move_tapes=True, rename_5_6=''):
    """
    Runs the LBLRTM executable.

    output_path:    path to a directory where LBLRTM output will be stored
    file_write:     if passed as a string, string will be written as an info file in the output_path directory. Ignored if False
    lblrtm_path:    path to directory containing LBLRTM executable and files
    lbl_exe_name:   file name of compiled LBLRTM executable
    move_tapes:     if True, will move TAPE[5,6,7,11,12] files to output_path. Recommended to keep this on.
    rename_5_6:     if set, this string will be appended to the TAPE5 and TAPE6 filenames to avoid overwriting
    """
    starting_directory = os.getcwd()
    os.chdir(lblrtm_path)

    print('Running LBLRTM')
    sub.call(lbl_exe_name)

    if move_tapes:
        util.move_files(['TAPE5', 'TAPE6', 'TAPE7', 'TAPE11', 'TAPE12',], 
                        output_path)
    
    if rename_5_6:
        os.rename('TAPE5', 'TAPE5'+rename_5_6)
        os.rename('TAPE6', 'TAPE6'+rename_5_6)

    if file_write and isinstance(file_write, str):
        with open(output_path + 'lblrtm_info.txt', 'w+') as ifile:
            ifile.write('Info from LBLRTM run on {:}:\n'.format(str(np.datetime64('now'))))
            ifile.write(file_write)
    
    os.chdir(starting_directory)


def write_emis_sfc_type(surface, lblrtm_path=_lblpath, source='/net/thunder/data1/bjp224/WHAFFFERS/emis/surface_emissivity_for_11types_0deg.nc'):

    sfc_type_list = ['grass', 'drygrass', 'deciduous', 'conifer', 'purewater', 'finesnow', 'mediumsnow', 'coarsesnow', 'ice']

    sfc_tag_list = ['grs', 'dgr', 'dcd', 'cnf', 'pwa', 'fsn', 'msn', 'csn', 'ice']

    num = np.arange(len(sfc_type_list))

    sfc_dict = dict(zip(sfc_type_list, num))
    sfc_tag_dict = dict(zip(sfc_type_list, sfc_tag_list))

    surface_lc = re.sub('[, _-]', '', surface).casefold()
    
    try:
        sfc_index = sfc_dict[surface_lc]
    except KeyError:
        raise Exception(f'Bad surface type specification: {surface}')
    
    emiss_dataset = xr.load_dataset(source)

    sfc_emiss = emiss_dataset.emis_sim[sfc_index, :]

    wn, emis = sfc_emiss.wn.values, sfc_emiss.values

    write_lbl_emiss(lblrtm_path, wn, emis)

    print(f'Emissivity and reflectivity files written for {surface} surface')

    return '.' + sfc_tag_dict[surface_lc]



"""WRITE_LBL_EMISS

File to write EMISSIVITY and REFLECTIVITY files needed for simulating
upwelling radiation
Author: Sanjeevani Panditharatne
"""
import numpy as np
def write_lbl_emiss(lbl_location,wn,emiss):
    """Function to automatically generate EMISSIVITY and REFLECTIVITY files

    Args:
        lbl_location (str): Path of LBLRTM exe
        wn (np.array): Wavenumber range
        emiss (np.array): Emissivity values
    """
    refl=np.array([1-x for x in emiss])

    with open(lbl_location+'EMISSIVITY', "w+") as file:
        # Record 1.4 
        # V1,V2,DV,NLIM
        file.write("{:10.4f}{:10.4f}{:10.4f}     {:5d}\n".format(wn[0],wn[-1],wn[1]-wn[0],len(wn)))
        for j in range(len(wn)):
            file.write("{:10.7f}{:10.4f}\n".format(emiss[j], wn[j]))

    with open(lbl_location+'REFLECTIVITY', "w+") as file:
        # Record 1.4 
        # V1,V2,DV,NLIM
        file.write("{:10.4f}{:10.4f}{:10.4f}     {:5d}\n".format(wn[0],wn[-1],wn[1]-wn[0],len(wn)))
        for j in range(len(wn)):
            file.write("{:10.7f}{:10.4f}\n".format(refl[j], wn[j]))

    return

def read_config_file(config_filepath):
    """
    Read a config file at config_filepath with lines in the format:
    KEYWORD     value   # comment denoted by # or ;
    and convert it to a dictionary {KEYWORD:value}
    Attempts to load in numbers as floats or ints.
    """

    config_dict = {}
    with open(config_filepath) as f:
        for line in f:
            # remove newlines
            line = line.replace('\n', '')
            # trim comments using # or ;
            line = line.split('#')[0]
            line = line.split(';')[0]
            # remove leading/trailing whitespace
            line = line.strip()
            # reject if empty
            if line != '':
                # separate values
                line_values = line.split()
                if len(line_values) == 2:
                    opt_key, opt_value = line_values
                else:
                    raise ValueError(f'Line not formatted correctly: {line}')
                # convert numbers to strings if possible
                try:
                    if '.' in opt_value:
                        opt_value = float(opt_value)
                    else:
                        opt_value = int(opt_value)
                except ValueError:
                    pass
                config_dict[opt_key] = opt_value
    return config_dict



def get_lblrtm_config(config_fp, **kwargs):
    """
    Loads in configuration info from config_fp and merges with any kwargs submitted.
    kwargs submitted with LBLRTM variable names will overwrite any values from the config file.
    kwargs submitted with aliases will overwrite both of the above.

    see alias_dict for a full list of aliases available, that are hopefully more descriptive than LBLRTM!
    """
    
    config_info = read_config_file(config_fp)

    alias_list = alias_dict.keys()

    kwargs_alias = {key:itm for (key,itm) in kwargs.items() if key in alias_list}
    kwargs_lblrtm = {key:itm for (key,itm) in kwargs.items() if key not in alias_list}

    if 'wn_range' in kwargs:
        wn_range = kwargs['wn_range']
        kwargs_alias['min_wn'] = wn_range[0]
        kwargs_alias['max_wn'] = wn_range[1]

    # with this line, any variables submitted as LBLRTM kwargs will overrule config file options
    config_info = config_info | kwargs_lblrtm

    # variables submitted as aliases add to or overwrite previously defined values
    for key, itm in kwargs_alias.items():
        if key not in ['surf_emis_coeffs', 'surf_refl_coeffs']:
            config_info[ alias_dict[key] ] = itm

    # handling array input of surface emissivity/reflectivity coefficients
    for prop in ['EMIS', 'REFL']:
        alias_key = f'surf_{prop.casefold()}_coeffs'

        if alias_key in kwargs_alias:
            prop_values = kwargs_alias[alias_key]
            
            if len(prop_values) < 3:
                for n, value in enumerate(prop_values):
                    config_info[f'SR{prop}({n+1})'] = value
            else:
                raise ValueError(f'Too many values in specification of surface property {prop}')
    
    return config_info


def get_config_options(var_list, config_dict, **kwargs):
    """
    For a list of variables (LBLRTM names) in var_list, return the values in the config_dict.
    Order of variables will be preserved.
    """

    options = []
    for variable in var_list:
        try:
            options += [config_dict[variable]]
        except KeyError:
            raise Exception(f'No value found for {variable} in config file or keyword args')
    
    return options


def items_from_config(config_dict, var_list):
    """
    For a list of variables (LBLRTM names) in var_list, return the values in the config_dict.
    Order of variables will be preserved.
    """
    return [config_dict[var_name] for var_name in var_list]



def merge_gas_profiles(species_dict, n_lvls, xsections=False, model_atm=0, n_gases_min=7,):
    """
    Merge the gases in species_dict into a matrix of shape (n_lvls, n_species) for writing
    into LBLRTM TAPE5. Units info is also parsed and returned as output.

    species_dict:   dict with entries {'GAS_NAME':{*further info*}, }
    n_lvls:         number of atmospheric levels in LBLRTM simulation
    xsections:      set as True if using LBLRTM 'cross-section' gases
    model_atm:      set as 0-6 to access the default LBLRTM profiles for unspecified gases
                    (0 = not simulated)
    n_gases_min:    minimum number of gases to simulate (in order from Table I of instructions)
    """

    if not xsections:

        biggest_number_gas = max(gas_number_dict[gas] for gas, _ in species_dict.items())
        n_species = max(biggest_number_gas, n_gases_min)

        profile_mat = np.zeros((n_lvls, n_species))

        units_list = [str(model_atm)] * n_species

        for gas, info_dict in species_dict.items():

            number = gas_number_dict[gas]

            if 'profile' in info_dict:
                profile_mat[:, number-1] = info_dict['profile']

            elif 'amount' in info_dict:
                profile_mat[:, number-1] = np.full(n_lvls, info_dict['amount'])

            if 'unit' in info_dict:
                units_list[number-1] = info_dict['unit']
            elif 'units' in info_dict:
                units_list[number-1] = info_dict['units']
            else:
                print(f'No unit specified for {gas} amount, assuming default units (ppmv)')
                units_list[number-1] = ' '

        units_string = ''.join(units_list)
        return profile_mat, units_string
    
    else:
        
        n_species = len(species_dict)

        units_list = [' '] * n_species

        profile_mat = np.zeros((n_lvls, n_species))

        for n, (gas, info_dict) in enumerate(species_dict.items()):

            if 'profile' in info_dict:
                profile_mat[:, n] = info_dict['profile']

            elif 'amount' in info_dict:
                profile_mat[:, n] = np.full(n_lvls, info_dict['amount'])

            if 'unit' in info_dict:
                units_list[n] = info_dict['unit']
            elif 'units' in info_dict:
                units_list[n-1] = info_dict['units']
            else:
                print(f'No unit specified for {gas} amount, assuming default units (ppmv)')
                units_list[n] = ' '

            if gas not in xsect_gas_list:
                print(f'Warning: {gas} not in set of available cross-section gases')

        units_string = ''.join(units_list)
        return profile_mat, units_string




def write_lblrtm_tape5(tape5_fp=_lblpath+'TAPE5', config_fp=_lblpath+'lblrtm_config', profile_dict=None, gas_input_dict=None, use_pressure_coords=True,
                       profile_description='', xsect_input_dict=None, xsect_prfl_description='', atm_gas_defaults=6, **kwargs):
    """
    Write a formatted LBLRTM input (TAPE5) text file to the specified filepath.

    tape5_fp:               Destination filepath for tape5 file (should end with 'TAPE5')
    config_fp:              Filepath to LBLRTM config text file
    profile_dict:           Dict containing {'pressure':[], 'temperature':[], 'altitude':[]} (or similar) for the profile
                                NB profile must be strictly in order of increasing altitude/decreasing pressure.
    gas_input_dict:         Dict with entries {'GAS_NAME':{further info}, }
    use_pressure_coords:    Use pressure as principal vertical coordinate if set to True, otherwise altitude.
    profile_description:    String describing profile which will be written on TAPE5 (24 char limit)
    xsect_input_dict:       Dict with entries {'XS_GAS_NAME':{further info}, }
    xsect_prfl_description: String describing cross-section data which will be written on TAPE5 (50 char limit)
    atm_gas_defaults:       LBLRTM preset atmosphere to be selected for unspecified gases (1-6, see instructions)

    kwargs:         keyword arguments input either as LBLRTM variable names (see instructions) or aliases (see alias_dict)
    """

    pyfile_name = 'lblrtm.py'

    cfg = get_lblrtm_config(config_fp, **kwargs)

    # Record 1.2
    hirac_opt, lblf4_opt, continuum_opt, aerosol_opt, calc_result_opt, scan_result_opt, = get_config_options(
        ['IHIRAC', 'ILBLF4', 'ICNTNM', 'IAERSL', 'IEMIT', 'ISCAN',], cfg)
    filter_opt, plot_opt, test_opt, lblatm_opt, merge_opt, optical_depth_opt, xsect_opt = get_config_options(
        ['IFILTR', 'IPLOT', 'ITEST', 'IATM', 'IMRG', 'IOD', 'IXSECT'], cfg)

    if cfg['IATM']:
        if type(profile_dict['altitude']) == float:
        # catching case where only surface altitude specified
            altitude_array = np.zeros_like(profile_dict['pressure'])
            altitude_array[0] = profile_dict['altitude']
            altitude = altitude_array
        elif len(profile_dict['altitude']) == 1:
            altitude_array = np.zeros_like(profile_dict['pressure'])
            altitude_array[0] = profile_dict['altitude'][0]
            altitude = altitude_array
        else:
            altitude = profile_dict['altitude']

        pressure = profile_dict['pressure']
        temperature = profile_dict['temperature']

        # allows switching of vertical coordinates from pressure to altitude
        if use_pressure_coords:
            vertical_coord_flag = -1
            vertical_coord = pressure
        else:
            vertical_coord_flag = 1
            vertical_coord = altitude

        n_lvls = len(vertical_coord)

        cfg['IBMAX'] = vertical_coord_flag * n_lvls
        
        jchar_p, jchar_t, j_long = get_config_options(['JCHARP', 'JCHART', 'JLONG'], cfg)

        if j_long == 'L':
            gas_wrt_fmt = '15.8E'
        else:
            gas_wrt_fmt = '10.3E'

        gas_profiles, jchar_molecules = merge_gas_profiles(gas_input_dict, n_lvls, model_atm=atm_gas_defaults, n_gases_min=cfg['NMOL'])

        cfg['NMOL'] = len(jchar_molecules)  # overwrite with the number of molecules actually selected

    if xsect_input_dict:
        cfg['IXSECT'] = 1

    if cfg['IXSECT']:

        # ALLOW IPRFL OPTION
        xsect_user_profile, xsect_p_convol = get_config_options(['IPRFL', 'IXSBIN'], cfg)
        
        # n_lvls_xsect = 
        if 'coord' in xsect_input_dict:
            raise NotImplementedError()
        # ALLOW SEPARATE LEVEL SPECIFICATION
        else:
            n_lvls_xsect = n_lvls
            vertical_coord_xsect = vertical_coord
            vertical_coord_flag_xsect = {1:0, -1:1}[vertical_coord_flag]

        xsect_species = list(xsect_input_dict.keys())

        n_xsect_molecules = len(xsect_input_dict)

        xsect_gas_profiles, jchar_xsect_molecules = merge_gas_profiles(xsect_input_dict, n_lvls_xsect, xsections=True)

    # TODO: eliminate most instances of get_config_options() and replace with items_from_config() in writing stage
    # TODO: maybe also merge contents of the above functions

    # ----------------------------------------- WRITING TAPE5 FILE -----------------------------------------

    with open(tape5_fp, "w+") as file:

        # Record 1.1       User Identification
        # CXID:  80 characters of user identification  (80A1)
            # BP NOTE $ is important to initialise file, and file is terminated with a %, otherwise will loop
        file.write(f"$ TAPE5 generated by write_lblrtm_tape5() in {pyfile_name} on {dt.datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}\n")

        # Record 1.2       LBLRTM Control Variables
        # IHIRAC, ILBLF4, ICNTNM, IAERSL,  IEMIT,  ISCAN, IFILTR, IPLOT, ITEST,  IATM,  IMRG,  ILAS,   IOD, IXSECT,  MPTS,  NPTS
        #      5,     10,     15,     20,     25,     30,     35,    40,    45,    50, 54-55,    60,    65,     70, 72-75, 77-80
        #  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1, 4X,I1, 4X,I1, 4X,I1, 3X,A2, 4X,I1, 4X,I1,  4X,I1, 1X,I4, 1X,I4 
        file.write(' HI={:1d} F4={:1d} CN={:1d} AE={:1d} EM={:1d} SC={:1d} FI={:1d} PL={:1d}'.format(
            hirac_opt, lblf4_opt, continuum_opt, aerosol_opt, calc_result_opt, scan_result_opt, filter_opt, plot_opt,
        ) +        ' TS={:1d} AM={:1d}MG={:2d} LA=0 OD={:1d} XS={:1d}    0    0\n'.format(
            test_opt, lblatm_opt, merge_opt, optical_depth_opt, xsect_opt
        ))

        if continuum_opt == 6:
            # Record 1.2.a        Continuum Factors         (required if ICNTNM = 6)
            # XSELF, XFRGN, XCO2C, XO3CN, XO2CN, XN2CN, XRAYL
            #  free format
            file.write(' {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n'.format(
                *items_from_config(cfg, ['XSELF', 'XFRGN', 'XCO2C', 'XO3CN', 'XO2CN', 'XN2CN', 'XRAYL'])
            ))


        if calc_result_opt == 2:
            # Record 1.2.1        LBL Solar Parameters      (required if IEMIT = 2; otherwise omit)
            #  INFLAG,  IOTFLG,  JULDAT
            #     1-5,    6-10,   13-15
            #      I5,      I5,  2X, I3
            # LOW PRIORITY FEATURE
            raise NotImplementedError()


        # NOTE code assumes the conditions on the next line will always be met, some use cases might not do this
        # Record 1.3          LBL Specifications        (required if IHIRAC > 0; IAERSL > 0; IEMIT = 1; IATM = 1; or ILAS > 0; otherwise omit)
        #      V1,     V2,   SAMPLE,   DVSET,  ALFAL0,   AVMASS,   DPTMIN,   DPTFAC,   ILNFLG,     DVOUT,   NMOL_SCAL
        #    1-10,  11-20,    21-30,   31-40,   41-50,    51-60,    61-70,    71-80,     85,      90-100,         105
        #   E10.3,  E10.3,    E10.3,   E10.3,   E10.3,    E10.3,    E10.3,    E10.3,    4X,I1,  5X,E10.3,       3x,I2
        file.write('{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}    {:1d}     {:10.3E}   {:2d}\n'.format(
            *items_from_config(cfg, ['V1', 'V2', 'SAMPLE', 'DVSET', 'ALFAL0', 'AVMASS', 'DPTMIN', 'DPTFAC', 'ILNFLG', 'DVOUT', 'NMOL_SCAL'])
        ))
            # min_wn, max_wn, samples_per_mean_hw, dv_setting, ave_coll_broad_hw, ave_mol_mass_doppler_hw, line_rejection_od, cont_od_rejection_factor,
            # line_rejection_recort_opt, dv_monochr_output_grid, n_molecules_scaled

        if cfg['NMOL_SCAL'] > 0:
            # Record 1.3.a        Profile Scaling Ctrl.     (required if NMOL_SCAL > 0;  otherwise omit)
            #   HMOL_SCAL(M=1,39)
            #               1-39
            #               39A1
            # MED PRIORITY FEATURE
            
            # TODO: implement logic of this.
            # NMOL_SCAL is largest molecule number to be scaled. user should input e.g. scaling_dict = {'h2o':{'type':'direct','factor':1.4}}
            # options are no scaling, scale directly, scale to molec/cm2 column amount, scale to DU col amount, scale to vmr, scale to pwv (cm)
            
            # below has 8 entries per line, up to NMOL_SCAL

            # Record 1.3.b.(1..n) Profile Scaling Values    (only if NMOL_SCAL > 0;  otherwise omit)
            #    (XMOL_SCAL, M=1,mol_max)
            #      (8E15.7)
            # MED PRIORITY FEATURE
            raise NotImplementedError()

        if calc_result_opt == 1 or calc_result_opt == 3:
            # Record 1.4          Boundary Parameters
            #  TBOUND, SREMIS(1), SREMIS(2), SREMIS(3), SRREFL(1), SRREFL(2), SRREFL(3), surf_refl
            #    1-10,     11-20,     21-30,     31-40,     41-50,     51-60,     61-70,    75  
            #   E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3    4X,1A
            file.write('{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}    {:1s}\n'.format(
                *items_from_config(cfg, ['TBOUND', 'SREMIS(1)', 'SREMIS(2)', 'SREMIS(3)', 'SRREFL(1)', 'SRREFL(2)', 'SRREFL(3)', 'surf_refl'])
            )) 

        if calc_result_opt == 3 and merge_opt in [40, 41, 42, 43]:
            # Record 1.5          Analytic Jacobian variable (required for Analytic Jacobian calculation: IEMIT=3 and IMRG=40,41,42 or 43)
            #  NSPCRT
            #     1-5
            #      I5
            file.write('{:5d}\n'.format(
                cfg['NSPCRT']
            ))

        if merge_opt in [35, 36, 40, 41, 45, 46]:
        # Record 1.6.a-d      Pathnames for OD files    (required if IMRG = 35-36, 40-41, 45-46; otherwise omit)
        #                                                     NOTE:  IMRG = 35-36, 40-41, 45-46 require separate optical depth files for each layer
        #    PTHODL, LAYTOT
        #     1-55,  57-60
        #      A55, 1X, I4
            file.write('{:<55s} {:4d}\n'.format(
                *items_from_config(cfg, ['PTHODL', 'LAYTOT'])
            ))
        # NOTE that LAYTOT is ignored for IEMIT [calc_result_opt] = 3 i.e. Jacobian case

        if lblatm_opt == 0:
            # Ignoring all options in Record 2.x (for no LBL Atmosphere)
            raise NotImplementedError('Writer only supports input using LBLATM routine; it is not currently possible to specify in Record 2.x')

        elif lblatm_opt == 1:
            # Record 3.1       LBLATM  (atmosphere)         records applicable if LBLATM selected (IATM=1)
            #   MODEL,  ITYPE, IBMAX,    ZERO,  NOPRNT,  NMOL, IPUNCH, IFXTYP,   MUNITS,    RE, HSPACE,  VBAR,      REF_LAT
            #       5,     10,    15,      20,      25,    30,     35,  36-37,    39-40, 41-50,  51-60, 61-70,        81-90
            #      I5,     I5,    I5,      I5,      I5,    I5,     I5,     I2,   1X, I2, F10.3,  F10.3, F10.3,   10x, F10.3
            file.write('{:5d}{:5d}{:5d}{:5d}{:5d}{:5d}{:5d}{:2d} {:2d}{:10.3f}{:10.3f}{:10.3f}          {:10.3f}\n'.format(
                *items_from_config(cfg, ['MODEL', 'ITYPE', 'IBMAX', 'ZERO', 'NOPRNT', 'NMOL', 'IPUNCH', 'IFXTYP', 'MUNITS', 'RE', 'HSPACE', 'VBAR', 'REF_LAT'])
            ))
                # internal_model_atm, path_type_opt, vertical_coord_flag * n_lvls, zero_small_abs, short_print_opt, n_species, layer_dat_write_opt,
                # tape7_control_opt, tape7_gas_unit_opt, earth_radius, space_altitude, refraction_freq, reference_latitude

            if cfg['ITYPE'] in [2,3]:
                # Record 3.2          Geometric Config, Slant path  (ITYPE = 2,3) (MODEL = 0-6)
                #      H1,    H2,   ANGLE,   RANGE,   BETA,   LEN,     HOBS
                #    1-10, 11-20,   21-30,   31-40,  41-50, 51-55,    61-70
                #   F10.3, F10.3,   F10.3,   F10.3,  F10.3,    I5, 5X,F10.3
                file.write('{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:5d}     {:10.3f}\n'.format(
                    *items_from_config(cfg, ['H1', 'H2', 'ANGLE', 'RANGE', 'BETA', 'LEN', 'HOBS'])
                ))
                    # obs_alt, end_or_tan_alt, zen_angle, path_length, path_angle, tan_height_opt, sat_obs_height



                # Record 3.3.a        Default Layering Spec.    For IBMAX  = 0 (from RECORD 3.1)
                # NOT USED
                if cfg['IBMAX'] == 0:
                    raise NotImplementedError('Writer only supports input user-defined layering; not currently possible to specify in Record 3.3.a')

                elif cfg['IBMAX'] != 0:
                    # Record 3.3.b        Level specification       For IBMAX != 0 (from RECORD 3.1)
                    #       For IBMAX > 0  (from RECORD 3.1)
                    #         ZBND(I), I=1, IBMAX   altitudes of LBLRTM layer boundaries
                    #        (8F10.3)
                    #       If IBMAX < 0
                    #       PBND(I), I=1, ABS(IBMAX) pressures of LBLRTM layer boundaries
                    #        (8F10.3)
                    # TODO generally making it less profile-oriented to handle single layer calculation
                    file.write( util.wrap_str( vertical_coord , write_fmt='10.3f', n_cols=8) )
                        # NB MUST END WITH NEWLINE

            if cfg['ITYPE'] == 1:
                raise NotImplementedError('Horizontal path not yet implemented')
                # Record 3.2H         Horizontal Path               (ITYPE = 1) (MODEL = 0-6)
                #      H1,      ,      ,  RANGEF
                #    1-10,                31-40
                #   F10.3,   10X,   10X,  F10.3
                # LOW PRIORITY FEATURE

                # RECORD 3.3  not used for this case

            if cfg['MODEL'] == 0:
                # Record 3.4          Profile Controls      User Defined Atmospheric Profile (MODEL = 0)
                #  IMMAX,   HMOD
                #      5,   6-29
                #     I5,    3A8
                file.write('{:5d}{:>24s}\n'.format(
                    vertical_coord_flag * n_lvls, profile_description
                ))

                for i in range(n_lvls):
                    # Record 3.5          Level input
                    #    ZM,    PM,    TM,    JCHARP, JCHART,   JLONG,   (JCHAR(M),M =1,39)
                    #  1-10, 11-20, 21-30,        36,     37,      39,     41  through  80
                    # E10.3, E10.3, E10.3,   5x,  A1,     A1,  1x, A1,     1x,    39A1

                    file.write('{:10.3f}{:10.3f}{:10.3f}     {:1s}{:1s} {:1s} {:<39s}\n'.format(
                        altitude[i], pressure[i], temperature[i], jchar_p, jchar_t, j_long, jchar_molecules
                    ))

                    # Record 3.6.1-N      Level Concentrations
                    #    VMOL(M), M=1, NMOL
                    #    8E10.3
                    #    VMOL(M) density of the M'th molecule in units set by JCHAR(K)
                    #    **NOTE** If JLONG=L, then VMOL(M) is in 8E15.8 format

                    file.write( util.wrap_str( gas_profiles[i,:], write_fmt=gas_wrt_fmt, n_cols=8 ) )
                        # NB MUST END WITH NEWLINE

                #  REPEAT records 3.5 and 3.6.1 to 3.6.N for each of the remaining IMMAX boundaries


            if xsect_opt:
                # Record 3.7          Cross Section Control     these records applicable if LBLATM selected (IATM=1)
                #                                               AND cross-sections ARE selected (IXSECT=1)
                #   IXMOLS,  IPRFL, IXSBIN
                #        5,     10,     15
                #       I5,     I5,     I5
                file.write('{:5d}{:5d}{:5d}\n'.format(
                    n_xsect_molecules, xsect_user_profile, xsect_p_convol
                ))

                # Record 3.7.1           Names
                #   XSNAME(I), I=1, IXMOLS
                #   (7A10,(/,8A10))
                file.write( util.wrap_str( xsect_species, write_fmt='10s', n_cols=8, row1_ncols=7 ) )


                # Record 3.8             Profile Control        (IPRFL = 0)
                # LAYX,  IZORP,  XTITLE
                #    5,     10,   11-60
                #   I5,     I5      A50
                file.write('{:5d}{:5d}{:>50s}\n'.format(
                    n_lvls_xsect, vertical_coord_flag_xsect, xsect_prfl_description
                ))


                for j in range(n_lvls_xsect):
                    # Record 3.8.1           Level Control
                    #  ZORP,  (JCHAR(K),K =1,28)
                    #  1-10,    16  through  50
                    # F10.3, 5X,           35A1
                    file.write('{:10.3f}     {:<35s}\n'.format(
                        vertical_coord_xsect[j], jchar_xsect_molecules
                    ))

                
                    # Record 3.8.2-N         Level amounts
                    #   DENX(K), K=1, IXMOLS
                    #   8E10.3
                    #   DENX(K) density of the K'th cross-section in units set by JCHAR(K)
                    file.write( util.wrap_str( xsect_gas_profiles[i,:], write_fmt='10.3E', n_cols=8 ) )

                # REPEAT records 3.8.1 to 3.8.N for each of the remaining LAYX boundaries


        # Ignoring all options in Record 4.x (for LOWTRAN Aerosol specification)
        if aerosol_opt != 0:
            raise NotImplementedError('Aerosol input not available in this reader')


        # Ignoring all options in Record 5.x (for non-implemented Laser options)


        # NOTE if developing, that beyond this point there are repeated variable names e.g. JEMIT, NPTS
        # so some workaround will be required with the specification input

        if merge_opt in list(range(13, 19)) + list(range(23,29)) + [35, 36]:
            raise NotImplementedError('Scan and filter merges not available in this reader')
        # Record 6.        Scan Merge                   ( for scanned sequential results; IMRG between 13 and 18; 35-36 )
        # Record 6.1    

        # Record 7.1       Filter Merge                 ( for sequential results with filter; IMRG between 23 and 28 )
        # Record 7.2
        # Record 7.3-N

        if cfg['ISCAN'] != 0:
            raise NotImplementedError('Scanning of output spectra not available in this reader')
        # Record 8.1       Scan Function                        (ISCAN = 1)

        # Record 9.1       Interpolation Function               (ISCAN = 2)
        #   DVO,     V1,     V2,  JEMIT,   I4PT,   IUNIT,  IFILST, NIFILS, JUNIT,  NPTS
        #  1-10,  11-20,  21-30,  31-35,  36-40,   56-60,  61-65,   66-70, 71-75, 76-80
        # F10.3,  F10.3,  F10.3,     I5      I5,  15X,I5,     I5,      I5,    I5,    I5
        # MED PRIORITY FEATURE
        
    
        # Record 10.1      FFT Scan                             (ISCAN = 3)
        # Record 10.2

        if cfg['IFILTR'] == 1:
            raise NotImplementedError('Filter function integration of spectra not available in this reader')
        # Record 11.1      Filter Function                      (IFILTR = 1)
        # Record 11.2
        # Record 11.3-N

        file.write("%%%%%%%%")



def match_units(unit_string):
    """
    Try to match unit description in unit_string to an LBLRTM JCHAR(M).
    """
    chars = 'ABCDE'
    potential_matches = [
        'volume mixing ratio',  # ppmv
        'number density',       # cm**-3
        'mass mixing ratio',    # g kg**-1
        'mass density',         # g m**-3
        'partial pressure'      # mbar (=hPa)
        ]

    potential_chars = [char for char, match_string in zip(chars, potential_matches) if match_string in unit_string]

    return potential_chars



def parse_profile_gas(profile_ds:xr.Dataset, target:str, xsect=False, verbose=False):
    """
    Attempts to parse an xarray.Dataset containing the atmospheric profile into a dictionary ready to pass to
    the TAPE5 writer.
    """

    # compiling list of potential keys for target gas in profile
    key_options = [target.casefold()]
    if target == 'H2O':
        key_options += ['q', 'w']
    elif target == 'O3':
        key_options += ['ozone']

    # checking if any are in the dataset
    target_in_ds_with_key = [key for key in key_options if key in profile_ds.variables]

    # ignore if none found
    if len(target_in_ds_with_key)==0:
        if verbose: print(f'{target} not found in profile dataset')
        pass

    else:
        key_to_use = target_in_ds_with_key[0]
        if verbose: print(f'{target} found in profile dataset with key {key_to_use}')

        gas_data = profile_ds[key_to_use]

        if gas_data.size > 1:
            # profile gas
            var_dict = {'profile':gas_data.values}

        elif gas_data.size == 1:
            # column amount gas
            var_dict = {'amount':gas_data.values}

        # trying to work out units
        if not xsect:
            if hasattr(gas_data, 'long_name'):
                potential_jchars = match_units(gas_data.long_name)
                
                if len(potential_jchars) == 1:
                    var_dict['units'] = potential_jchars[0]
                    if verbose: print(f'Unit {var_dict['units']} allocated for {target}')
                
            if not ('units' in var_dict.keys()):
                if hasattr(gas_data, 'units') or hasattr(gas_data, 'unit'):
                    # scan 'units' or 'unit' attribute of variables, if it features 'parts per million' or 'ppm'
                    # then assume it's volume mixing ratio
                    try: 
                        if 'parts per million' in gas_data.units or 'ppm' in gas_data.units:
                            var_dict['units'] = 'A'
                            if verbose: print(f'Unit {var_dict['units']} allocated for {target}')

                    except AttributeError:
                        if 'parts per million' in gas_data.unit or 'ppm' in gas_data.unit:
                            var_dict['units'] = 'A'
                            if verbose: print(f'Unit {var_dict['units']} allocated for {target}')

            if not ('units' in var_dict.keys()):
                # if nothing found after all these checks, just leave it as default
                var_dict['units'] = ' '
                print(f'Caution, No unit info found for {target}, leaving as default unit (ppmv)')

        else:
            # only option for cross-section molecules
            var_dict['units'] = 'A'
            if verbose: print(f'Unit {var_dict['units']} allocated for {target}')

        return var_dict



def make_profile_dicts(profile:xr.Dataset, change_input_keys:dict={}):
    """
    This function will try to make a set of input dicts to write_lblrtm_tape5 using the provided xarray Dataset.
    It should be able to detect which gases are passed as column quantities and which are profile, and will make
    a guess about the units based off attributes of the xarray profile Dataset.
    Note that it will NOT attempt to convert units so ensure they are in the appropriate LBLRTM quantities for this.
    (See table I of the LBLRTM instructions)
    
    In the Dataset, name gas variables as their chemical symbols, (or q, w for H2O), and include one of 
    'volume mixing ratio', 'number density', 'mass mixing ratio', 'mass density', 'partial pressure'
    in the .long_name variable attribute. Otherwise, including 'parts per million' or 'ppm' in the .units attribute will allow
    the code to assume you are talking about volume mixing ratio. Failing that, the code will set to default unit (ppmv).

    change_input_keys argument allows to set e.g. 'pressure' to be 'pressure_level' in the input profile dataset
        i.e. change_input_keys = {'pressure':'pressure_level', 'altitude':'z'}
    """

    input_map = {
        'pressure':'p',
        'temperature':'t',
        'altitude':'altitude',

        # 'H2O':'q',
        # 'CO2': 'co2',
        # 'O3': 'o3',
        # 'N2O':'n2o',
        # 'CO': 'co',
        # 'CH4': 'ch4',
        # 'SF6':'sf6',

        # 'CCL4':'ccl4',
        # 'CFC11':'cfc11',
        # 'CFC12':'cfc12',
    }

    keys = input_map | change_input_keys

    prof_dict = {variable: profile[keys[variable]].values for variable in ['pressure','temperature','altitude']}

    # profile_gases = ['H2O', 'O3', 'CO', 'CO2', 'CH4']
    # amount_gases = ['N2O', 'SF6']
    # xsect_gases = ['CCL4', 'CFC11', 'CFC12']

    # # gases supplied in profile
    # gas_dict = {variable: {'profile':profile[keys[variable]].values,
                            #  'unit':'C'} for variable in profile_gases}

    gas_dict = {gas: parse_profile_gas(profile, gas) for gas in gas_list if parse_profile_gas(profile, gas)}

    xsect_dict = {gas: parse_profile_gas(profile, gas, xsect=True) for gas in xsect_gas_list if parse_profile_gas(profile, gas, xsect=True)}
    # # gases supplied as single column amount
    # for variable in amount_gases:
    #     gas_dict[variable] = {'amount': profile[keys[variable]].values, 'unit':'A'}
    #     # unit A means values as volume mixing ratio = ppmv

    # try:
    #     xsect_dict = {variable: {'amount':profile[keys[variable]].values,
    #                              'units':'A'} for variable in xsect_gases}
    #     # unit A means values as volume mixing ratio = ppmv

    # except KeyError:
    #     xsect_dict = {}

    return prof_dict, gas_dict, xsect_dict


def slant_path_run(profile, wnum_range, angle, lbl_atm, sfc_type, save_dir, dw_obs_alt=0.130, profile_src='',
                   profile_txt='', xsect_prof_txt='', test_str='', sonde_fp='', lbl_exe_loc=_lblpath):
    """
    Run LBLRTM in 'slant path' mode (ITYPE=2) with an input profile.

    profile:    xarray.Dataset containing profile quantities
    wnum_range: wavenumber range for LBLRTM simulation
    angle:      viewing angle at observer, as offset from zenith (i.e. 0=looking up, 180=looking down)
    lbl_atm:    for gases left unspecified, the standard LBLRTM profile to be used
    sfc_type:   if angle > 90, (i.e. nadir-viewing from aircraft/space) will try to find a matching surface
    save_dir:   directory in which to save simulations (note that function will add to this to make a matching directory)
    dw_obs_alt: altitude (km) set for downwelling observations, should be height of radiometer above sea level
    profile_src:    source of the profile data
    profile_txt:    description
    xsect_prof_txt: description
    test_str:   string to be added to file names to indicate a test run
    sonde_fp:   radiosonde data filepath (for info only)
    lbl_exe_loc:    path to folder containing LBLRTM executable
    """


    # if/elif to handle different specifications of input profile
    if 'p_lev' in profile_src.casefold():

        make_dict_mod = {'altitude':'z', 'pressure':'pressure_level'}
        info_keys = ['valid_time', 'latitude', 'longitude']
        p_bound_key = 'pressure_level'
        sfc_index = np.argmin(np.abs(profile['z'].values-dw_obs_alt))
        p_surf_obs = profile['pressure_level'].values[sfc_index]
        sonde_info = ''
        src_tag = 'ERA5_pl'

    elif 'era' in profile_src.casefold():
        make_dict_mod = {}
        if 'follow' in profile_src.casefold():
            p_bound_key = 'p'
            info_keys = ['era5_time', 'era5_latitude', 'era5_longitude']
        else:
            info_keys = ['time', 'latitude', 'longitude']
            p_bound_key = 'p_half'
        sfc_index = np.argmin(np.abs(profile['altitude'].values-dw_obs_alt))
        p_surf_obs = profile['p'].values[sfc_index]
        sonde_info = ''
        src_tag = 'ERA5'

    elif 'sonde' in profile_src.casefold():
        make_dict_mod = {'altitude':'z'}
        info_keys = ['era5_time', 'era5_latitude', 'era5_longitude']
        p_bound_key = 'p'
        sonde_info = f'Radiosonde launch on the WHAFFFERS campaign, at file {sonde_fp}'
        sfc_index = np.argmin(np.abs(profile['z'].values-dw_obs_alt))
        p_surf_obs = profile['p'].values[sfc_index]
        if 'hybrid' in profile_src.casefold():
            src_tag = 'sonde_sentinel'
            # p_surf_obs = np.max(profile['p'].values)
        else:
            src_tag = 'sonde'
            # p_surf_obs = np.max(profile['sonde_bounds_press'].values)

    info_key_dict = dict(zip(['time','lat','lon'],info_keys))

    print(f'LBLRTM run in vertical mode with angle {angle:.2f} degrees')

    if angle < 90:
        # downwelling case
        welling_tag = "dw"
        obs_lvl = p_surf_obs
        if 'era' in profile_src.casefold() and 'p_lev' not in profile_src.casefold():
            profile = profile.isel(level=slice(sfc_index,None), half_level=slice(sfc_index,None))

        start_lvl = max(profile[p_bound_key].values.min(), 0.001) # LBLRTM input format restricts this
        t_boundary = 2.7 # K, outer space
        sfc_type = None
        print('Viewing geometry: downwelling (upwards-viewing), with space blackbody background')

    elif 90 < angle <= 180:
        # upwelling case
        welling_tag = f"uw.{180-angle:.2f}deg"
        obs_lvl = 0.001  # hPa, obs pressure 
        start_lvl = profile[p_bound_key].values.max()
        t_boundary = profile.skt.values
        print('Viewing geometry: upwelling (downwards-viewing)')

    else:
        raise Exception(f'Bad angle specification: {angle}deg')
    
    if sfc_type:
        # using specified surface type
        sfc_tag = write_emis_sfc_type(sfc_type, lblrtm_path=lbl_exe_loc)
        print(f'Using {sfc_type} surface type for IR emissivity')
        surface_info = f'Using {sfc_type} surface type for IR emissivity from file: surface_emissivity_for_11types_0deg.nc'
        emis_coeff=[-1.]
        refl_coeff=[-1.]
    else:
        # using blackbody surface
        surface_info = ''
        sfc_tag = ''
        emis_coeff=[1.]
        refl_coeff=[0.]

    profile = profile.assign({'obs_height': (obs_lvl)})
    profile['obs_height'] = profile['obs_height'].assign_attrs({'long_name':'observation pressure level passed to LBLRTM',
                                                                'unit':'hPa',})

    profile_input, gas_input, xsect_input = make_profile_dicts(profile, change_input_keys=make_dict_mod)

    using_xsect = len(xsect_input) > 0

    write_lblrtm_tape5(
        tape5_fp=lbl_exe_loc+'TAPE5',
        config_fp=lbl_exe_loc+'lblrtm_config',

        profile_dict=profile_input,
        gas_input_dict=gas_input,
        profile_description=profile_txt,
        xsect_opt=using_xsect,
        xsect_input_dict=xsect_input,
        xsect_prfl_description=xsect_prof_txt,
        atm_gas_defaults=lbl_atm,

        wn_range=wnum_range,
        boundary_temp=t_boundary,
        obs_alt=obs_lvl,
        end_or_tan_alt=start_lvl,
        zen_angle=angle,
        surf_emis_coeffs=emis_coeff,
        surf_refl_coeffs=refl_coeff,
    )
    
    info_string = ('Profile:\n'
                + sonde_info
                + f'Coordinates: ({profile[info_key_dict['lat']].data:.2f}, {profile[info_key_dict['lon']].data:.2f}), time: {profile[info_key_dict['time']].data}\n'
                + 'Data source: ERA5 complete (on model levels)\n'
                + 'Using variables T, q, o3, lnsp, z, and skt (for upwelling). lnsp used to calculate pressures of model levels.\n'
                + 'z converted from geopotential to altitude to use as basis for hydrostatic relation.\n'
                + f'LBLRTM parameters: observation angle {angle:.2f} deg, boundary temp {t_boundary:.2f} K, model atmosphere {lbl_atm}\n'
                + f'Surface altitude {dw_obs_alt:.2f} km & pressure {p_surf_obs:.1f} hPa (for dw obs only)\n'
                + f'GHGs added as detailed in profile.nc, from a range of sources (CAMS GHG forecasts, climate.gov, NOAA) \n'
                + surface_info)

    out_dir = util.gen_output_directory(str(profile[info_key_dict['time']].values), profile[info_key_dict['lat']].values, profile[info_key_dict['lon']].values)
    output_loc = save_dir + test_str + src_tag + out_dir + welling_tag + sfc_tag + '/'

    lbl_start_time = time.time()
    run_LBLRTM(output_loc, file_write=info_string)
    lbl_end_time = time.time()

    profile.to_netcdf(output_loc + 'profile.nc')

    print(f'LBLRTM finished with runtime {lbl_end_time - lbl_start_time:.1f}s, and profile saved')