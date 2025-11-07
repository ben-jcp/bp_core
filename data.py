import numpy as np
import xarray as xr
import datetime as dt
import re, os, time, cdsapi
from netCDF4 import Dataset
from . import ceda_io, util

def gridpoint_select(dataset, time, lat, lon,):
    # BP 04Nov2024
    # would be good for this to work for ranges in all dims
    # would be good for it to work with a range of geospatial data (common need)

    filtered_dataset = dataset.sel(
        valid_time=np.datetime64(time),
        latitude=lat,
        longitude=lon,
        method='nearest'
    )
    
    print(f'Closest gridpoint selected: (lat,lon): ({filtered_dataset.latitude.data:.2f},'
          +f'{filtered_dataset.longitude.data:.2f}), time: {str(filtered_dataset.valid_time.data)[:16]}')

    return filtered_dataset


def spatial_filter(dataset, params):
    """
    params  :   if length 2, then (lat,lon) of location to find (nearest point with tolerance 0.5)
                if length 4, then (W, E, S, N) boundaries of location 
    """

    if np.min(dataset.longitude.values) < 0:
        # longitude is not positive-definite, i.e. -180 to 180
        lon_nat = True
    else:
        # longitude is positive definite
        lon_nat = False

    if len(params) == 2:

        lat, lon = params

        if not lon_nat:
            lon = np.mod(lon, 360)

        filtered_dataset = dataset.sel(
            latitude=lat,
            longitude=lon,
            method='nearest',
            tolerance=0.5
        )

    elif len(params) == 4:

        min_lon, max_lon, min_lat, max_lat = params

        if not lon_nat:
            min_lon, max_lon = np.mod([min_lon, max_lon], 360)

        if min_lon > max_lon:
            dataset_1 = dataset.sel(
                latitude=slice(max_lat, min_lat),
                longitude=slice(min_lon, 360.)
            )
            dataset_2 = dataset.sel(
                latitude=slice(max_lat, min_lat),
                longitude=slice(-180., max_lon)
            )
            filtered_dataset = xr.concat([dataset_1, dataset_2], dim='longitude')

        else:
            filtered_dataset = dataset.sel(
                latitude=slice(max_lat, min_lat),  # ordering like this as ERA5 data in order of decreasing lat
                longitude=slice(min_lon, max_lon)
            )
    
    else:
        raise Exception('Incorrect spatial filtering parameter specification')
    
    return filtered_dataset


def extract_attrs(dataset, key, replace=None, excl_GRIB=True):
    if excl_GRIB:
        attrs_tocopy = {val:desc for val,desc in dataset[key].attrs.items() if 'GRIB' not in val
                        and val!='coordinates'}
    else:
        attrs_tocopy = dataset[key].attrs.items()

    if replace:
        for val, desc in replace.items():
            attrs_tocopy[val] = desc
            # can this be done as some kind of list comp?

    return attrs_tocopy


def add_units(dataset, unit_pairs):
    for key in list(unit_pairs.keys()):
        new_dataArray = dataset[key].assign_attrs(extract_attrs(dataset,key,replace={'units':unit_pairs[key]}))
        dataset[key] = new_dataArray
    return dataset


def add_attrs(dataset, attr_dict):
    """
    attr_dict: key, item pairs:
        key = variable name (str)
        item = dict of attribute names (str): values
    """
    for variable in list(attr_dict.keys()):
        if variable in list(dataset.keys()):
            new_dataArray = dataset[variable].assign_attrs(attr_dict[variable])
            dataset[variable] = new_dataArray
    return dataset


def ncReport(ncFile):
    dims = ncFile.dimensions
    
    vars = ncFile.variables
    attributeNames = ['name','long_name','units','dimensions','shape',]

# use '\t'.join() method on functions

    print('DIMENSIONS')
    print('\t'.join(attributeNames))
    for dim in dims:
        attrResults = []
        for attribute in attributeNames:
            try:
                attrResults += [str(getattr(dims[dim],attribute))]
            except AttributeError:
                try:
                    attrResults += [str(getattr(vars[dim],attribute))]
                except (AttributeError, KeyError):
                    attrResults += ['']
        # print(attrResults)
        print('\t'.join(attrResults))

    print('VARIABLES')
    for var in vars:
        if var not in dims:
            attrResults = []
            for attribute in attributeNames:
                try:
                    attrResults += [str(getattr(vars[var],attribute))]
                except (AttributeError, KeyError):
                    attrResults += ['']
            print('\t'.join(attrResults))


def find_slice(array, start, end, array_r=None):
    if array_r is None:
        array_r = array

    # NB GT/LT here is giving off by one errors of some sort but carrying on for now
    start_idx = max(np.argwhere(array_r >= np.round(float(start),4)).min(),0)
    end_idx = np.argwhere(array <= np.round(float(end),4)).max()+1

    return slice(start_idx,end_idx)


def coord_convert_from_str(coord_str):
    """
    https://stackoverflow.com/questions/33997361/how-to-convert-degree-minute-second-to-degree-decimal
    """
    deg, minutes, seconds, direction =  re.split('[°\'"]', coord_str)
    coord = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)

    return coord


def netcdf_to_xarray(filepath):
    """
    Converts a netCDF file to an xarray Dataset object. This is usually automatic but some files seem to upset xarray.
    Does not handle variables kept in groups.

    Arguments:
        :filepath: filepath of the netCDF dataset to load
    Returns:
        :ds: xarray Dataset containing all root-level data and attributes of the netCDF file
    """
    # load netCDF file with netcdf4 Python library, initialise xarray object
    ncf = Dataset(filepath)
    ds = xr.Dataset()

    # copy all general attributes to netCDF file
    ds = ds.assign_attrs( {attr:ncf.getncattr(attr) for attr in ncf.ncattrs()} )

    # copy all dimensions and their size to xarray object
    for dim_name in ncf.dimensions.keys():
        ds = ds.expand_dims( {dim_name:ncf.dimensions[dim_name].size} )        
    
    # copy all variables and their data across to xarray object, as well as their attributes (including units)
    for var in ncf.variables.keys():
        ds = ds.assign( {var:(ncf.variables[var].dimensions, ncf.variables[var][:])} )
        ds[var] = ds[var].assign_attrs( {attr:ncf.variables[var].getncattr(attr) for attr in ncf.variables[var].ncattrs()} )

    ncf.close()

    return ds



def dataset_selection(dataset, labels_and_ranges):
    # want to filter dataset from range of criteria, not necessarily coordinates
    # in_range for each variable, and also get variable dimensions
    # select by dimensions, using indices featuring in all variable-range combos
    raise NotImplementedError
    # return dataset


def convert_geoms_time(dataset):
    
    time_lbl = 'DATETIME'
    if dataset[time_lbl].dtype == np.dtype('float64'):
        ds_times = util.epoch_time_conversion(dataset[time_lbl].values)
    elif dataset[time_lbl].dtype == np.dtype('S256'):
        ds_times = util.byte_timestamp_conversion(dataset[time_lbl].values)
    else:
        raise Exception('Dataset time labelling not recognised')

    # make 'time' the main time coordinate, and remove DATETIME
    dataset = dataset.assign({'time':([time_lbl], ds_times)}).swap_dims({time_lbl:'time'}).drop_vars([time_lbl])

    return dataset


def dataset_time_subset(dataset, centre_time, time_window_width, time_label='time'):

    if type(centre_time) == str:
        centre_time = np.datetime64(centre_time)

    if type(time_window_width) == int or type(time_window_width) == float:
        time_range = np.timedelta64(int(time_window_width), 's')

    subset = dataset.sel({time_label : slice(centre_time-time_range, centre_time+time_range)})

    return subset


def nondim_interp(dataset, interp_values, interp_key, var_keys, flip=True):

    dataset = dataset.sortby(interp_key, ascending=True)
    interp_values = np.sort(interp_values)

    results = []
    for var in var_keys:
        
        interped_var = np.interp(interp_values, dataset[interp_key], dataset[var])
        if flip:
            interped_var = np.flip(interped_var)

        results += [interped_var]

    return results


def open_era_file(path, source='disk', token=None):

    if source == 'disk':

        return xr.load_dataset(path, decode_timedelta=True)
    
    elif source == 'ceda':

        return ceda_io.open_ceda_dataset(path, token)
    
    else:

        raise Exception('Invalid specification of source for ERA filepath')
    

def dataset_load_and_process(path, source, token, filter, profile_datetime):
    
    ds = open_era_file(path, source, token)
    ds = spatial_filter(ds, filter)

    # select single time value and drop time dimension
    # and drop extra level dimension
    if source == 'disk' and 'cams' not in path:
        try:
            ds = ds.squeeze().sel(valid_time=[profile_datetime], method='nearest')
        except KeyError:
            ds = ds.sel(valid_time=[profile_datetime], method='nearest')
    elif 'cams' in path:
        lead_time_mins = int(profile_datetime[-2:]) + 60 * int(profile_datetime[-5:-3])
        lead_time = np.timedelta64(lead_time_mins, 'm')
        ds = ds.sel(forecast_period=[lead_time], method='nearest').squeeze()

    return ds


def era5_local_available(datetime_search, tag=None, filt_params=None):
    """
    W, E, S, N
    """
    try:
        ml_fp_fn = util.local_era_fp(datetime_search, 'an_ml', tag)
        sfc_fp_fn = util.local_era_fp(datetime_search, 'an_sfc', tag)

    except FileNotFoundError:
        # FNF error if year or month files not created
        return False

    try:
        met_vars = ['lnsp','z','t','q','o3']
        sl_vars = ['skt']
        truths = []

        for var in met_vars:
            ds = xr.load_dataset(ml_fp_fn(var))
            # check that filter parameters are within min/max lat/lon range
            truths += [check_within_range(ds, filt_params)]
        
        for var in sl_vars:
            ds = xr.load_dataset(sfc_fp_fn(var))
            # check that filter parameters are within min/max lat/lon range
            truths += [check_within_range(ds, filt_params)]

        if all(truths):
            return True
        else:
            allvars = np.array(met_vars + sl_vars)
            invalid_vars = allvars[np.logical_not(truths)]
            print('The following variables not available: ' + ', '.join(invalid_vars))
            return False

    except IndexError:
        # IndexError thrown by lambda fn if files not present for correct date/variables
        return False


def check_within_range(dataset, filter):
    # does not work over dateline for now

    if dataset.longitude.values.min() < 0 and dataset.longitude.values.max() < 180:
        # longitude is not positive-definite, i.e. -180 to 180
        lon_nat = True
    else:
        # longitude is positive definite
        lon_nat = False

    minlat, maxlat = dataset.latitude.values.min(), dataset.latitude.values.max()
    minlon, maxlon = dataset.longitude.values.min(), dataset.longitude.values.max()

    if len(filter) == 2:
        filt_lat, filt_lon = filter

        if not lon_nat:
            filt_lon = np.mod(filt_lon, 360)
            
        # checking selected point within dataset bounds
        lat_cond = minlat <= filt_lat and filt_lat <= maxlat
        lon_cond = minlon <= filt_lon and filt_lon <= maxlon

    elif len(filter) == 4:

        filt_min_lon, filt_max_lon, filt_min_lat, filt_max_lat = filter

        if not lon_nat:
            filt_min_lon, filt_max_lon = np.mod([filt_min_lon, filt_max_lon], 360)

        # checking selected region fully within dataset bounds
        # NB no check of completeness
        lat_cond = minlat <= filt_min_lat and filt_max_lat <= maxlat
        lon_cond = minlon <= filt_min_lon and filt_max_lon <= maxlon

    else:
        raise Exception('Incorrect spatial filtering parameter specification')
    
    return lat_cond and lon_cond


params_lookup = {'z':'129', # single level
                 't':'130',
                 'q':'133',
                 'lnsp':'152', # single level
                 'o3':'203',
                 'skt':'235', # surface
                 'fal':'243',  # surface
                 'clwc':'246',
                 'ciwc':'247',
                 'cc':'248',
                 'co2':'210061',
                 'ch4':'210062',
                 'co':'210123'}


def download_era5_ml(variables, date, times, levels, stream, product_type, area_NWSE=[], gridspace=0.25, tag=None):
    
    timelist = '/'.join([str(time) for time in times])
    if len(area_NWSE) == 4:
        area = '/'.join([str(round(coord)) for coord in area_NWSE])
    else:
        area = ''

    if type(gridspace) == float:
        grid = str(gridspace) + '/' + str(gridspace)
    else:
        grid = ''

    if type(levels) == str:
        levels = levels.casefold().strip()
        if levels == 'sfc' or levels == 'surface':
            level_list = '137'
        elif levels == 'all':
            level_list = '1/to/137'
        else:
            level_list = levels
            raise Warning('Unrecognised level speficiation')
    else:
        level_list = '/'.join([str(int(i)) for i in levels])

    sl_vars = [sl_var for sl_var in ['z', 'lnsp'] if sl_var in variables]
    sfc_vars = [sfc_var for sfc_var in ['skt', 'fal'] if sfc_var in variables]
    vars_3d = [var for var in variables if var not in sl_vars and var not in sfc_vars]

    if len(sl_vars) > 0:
        if product_type == 'es':
            lev_type = 'es'
        else:
            lev_type = 'ml_137'
        sl_var_path = gen_era_savepath(lev_type, date, tag, sl_vars)
        sl_var_codes = [params_lookup[var] for var in sl_vars]
        if len(sl_var_codes) == 1:
            sl_var_codes = sl_var_codes[0]
        download_era5_mars(sl_var_path, date, timelist, sl_var_codes, '1', 'ml', stream, product_type, area, grid)
        print('Single level data downloaded')
        time.sleep(0.5)

    if len(sfc_vars) > 0:
        if product_type == 'es':
            lev_type = 'es'
        else:
            lev_type = 'sl_1'
        sfc_var_path = gen_era_savepath(lev_type, date, tag, sfc_vars)
        sfc_var_codes = [params_lookup[var] for var in sfc_vars]
        if len(sfc_var_codes) == 1:
            sfc_var_codes = sfc_var_codes[0]
        download_era5_mars(sfc_var_path, date, timelist, sfc_var_codes, None, 'sfc', stream, product_type, area, grid)
        print('Surface level data downloaded')
        time.sleep(0.5)

    if len(vars_3d) > 0:
        if product_type == 'es':
            lev_type = 'es'
        else:
            lev_type = 'ml_137'
        var_3d_path = gen_era_savepath(lev_type, date, tag, vars_3d)
        var_3d_codes = [params_lookup[var] for var in vars_3d]
        if len(var_3d_codes) == 1:
            var_3d_codes = var_3d_codes[0]
        download_era5_mars(var_3d_path, date, timelist, var_3d_codes, level_list, 'ml', stream, product_type, area, grid)
        print('Full level data downloaded')
        time.sleep(0.5)


def gen_era_savepath(lev_type, date, tag, var_names):
    
    era5_base_fp = '/users/bjp224/era5/'

    date_obj = dt.datetime.fromisoformat(date)
    year = str(date_obj.year)
    month = str(date_obj.month).zfill(2)

    save_path = era5_base_fp + '/'.join([lev_type, year, month, date]) + '.' + '.'.join([tag]+var_names+['nc'])

    return save_path


def download_era5_mars(target_path, date, time, param, levelist, levtype, stream, prod_type, area, grid, format='netcdf'):

    dir_name = os.path.dirname(target_path)
    if not os.path.exists(dir_name):
        print(f'Creating directory at {dir_name}')
        os.makedirs(dir_name)

    request_info = {'date'      :   date,
                    'time'      :   time,
                    'param'     :   param,
                    'levtype'   :   levtype,
                    'stream'    :   stream,
                    'type'      :   prod_type,
                    'area'      :   area,
                    'grid'      :   grid,
                    'format'    :   format}
    
    if levelist:
        request_info['levelist'] = levelist

    c = cdsapi.Client()
    c.retrieve('reanalysis-era5-complete',
               request_info,
               target_path)
    


def load_radiosonde_nc(filepath):

    sonde_data = xr.load_dataset(filepath)

    if filepath.endswith(
        'balloon_sonde_mcgill_gault_20250121T205030Z_20250121T213020Z_001.nc'):
        filter_cond = np.logical_or(sonde_data.ALTITUDE.values < 8500., sonde_data.ALTITUDE.values >9097.)
        sonde_data = sonde_data.isel(ALTITUDE=filter_cond)
        sonde_data = sonde_data.isel(DATETIME=filter_cond)
        print('21 Jan 2025 radiosonde has erroneous data around 9km altitude that has been removed')

    sonde_data = convert_geoms_time(sonde_data)

    return sonde_data