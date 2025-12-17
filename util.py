"""
General utility functions for miscellaneous jobs.

Author: Ben Pery
Date:
"""
import numpy as np
import datetime as dt
import shutil
import os
import glob


def in_range(array, range, incl=False):
    """
    Returns a boolean array which is True for the members of array within the bounds specified by range.
    
    Arguments:
        :array:      numerical array
        :range:      list or tuple 
        :incl:       if True, inequalities are inclusive. Default is False
    Returns:
        numpy.ndarray 
    """

    if incl:
        return np.logical_and(range[0] <= array, array <= range[1])
    else:
        return np.logical_and(range[0] < array, array < range[1])


def move_files(filenames, dest_directory):
    """
    Moves the files referenced by filenames to dest_directory, creating the destination directory if necessary.

    Arguments:
        :filenames:         string or list of strings containing files to be moved 
        :dest_directory:    string specifying location to move files
    """
    if not dest_directory.endswith('/'):
        dest_directory += '/'
    
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        print('path created: {:}'.format(dest_directory))

    if isinstance(filenames, (str,)):
        filenames = [filenames]

    new_dirs = list(set([os.path.dirname(file) for file in filenames]))
    new_dirs_absolute = [os.path.join(dest_directory, new_dir) for new_dir in new_dirs]
    for new_dir in new_dirs_absolute:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print('path created: {:}'.format(new_dir))

    for fn in filenames:
        try:
            dst = shutil.move(fn, dest_directory+fn)
            print('file moved to {:}'.format(dst))
        except FileNotFoundError:
            print('no file {:} found to move'.format(fn))


def move_files_with_wildcard(file_type_list, src_dir, dst_dir):

    starting_directory = os.getcwd()
    os.chdir(src_dir)

    for pathtype in file_type_list:
        file_list = glob.glob(pathtype, recursive=False)
        move_files(file_list, dst_dir)

    os.chdir(starting_directory)


def gen_output_directory(datetime, lat, lon, pert_mag=0, pert_shape=None, pert_var=None, pert_extent=None):
    date = dt.datetime.fromisoformat(datetime)
    datestr = dt.datetime.strftime(date, '%Y%m%d')
    timestr = dt.datetime.strftime(date, '%H%M')

    if pert_mag != 0:
        pert_string = (f'p{pert_mag:.2f}.{pert_var}.{pert_shape}.' + 
                       f'{int(pert_extent[0]):02}.{int(pert_extent[1]):02}')
    else:
        pert_string = ''

    output_directory_name = '.'.join([datestr, timestr, f'{lat:02.1f}_{lon:02.1f}', pert_string])

    return output_directory_name


def ceda_era_fp(datetime_str, type_string):
    # type_string: 'an_ml' for model level data or 'an_sfc' for single level (e.g. skin temp)

    # round to nearest hour, as ERA data is hourly
    dt_obj = round_to_hours(dt.datetime.fromisoformat(datetime_str))

    yr_mo_day = dt_obj.strftime('%Y,%m,%d').split(',')

    # creating string for CEDA ERA5 archive 
    base_string = 'https://dap.ceda.ac.uk/badc/' + '/'.join(['ecmwf-era5','data','oper',type_string]+yr_mo_day) + \
        '/' + '_'.join(['ecmwf-era5', 'oper', type_string, dt_obj.strftime('%Y%m%d%H%M')]) + '.'
    
    return lambda var : base_string + var + '.nc'


def local_era_fp(datetime_str, type_string, tag):
    # round to nearest hour, as ERA data is hourly
    dt_obj = round_to_hours(dt.datetime.fromisoformat(datetime_str))

    level_code = {'an_sfc':'sl_1/', 'an_pl':'pl_37/',
                  'an_ml':'ml_137/', 'cams':'cams/', 'es':'es/'}

    yr_mo = dt_obj.strftime('%Y,%m').split(',')
    date_str = dt_obj.strftime('%Y-%m-%d')

    base_dir = '/users/bjp224/era5/'+level_code[type_string]+'/'.join(yr_mo)+'/'
    matching_files = [base_dir + file for file in os.listdir(base_dir) if date_str in file and tag in file]

    return lambda var : [var_file for var_file in matching_files if f'.{var}.' in var_file][0]


def any_era_fp(datetime_str, type_string, tag, source='disk'):

    if source == 'disk':

        return local_era_fp(datetime_str, type_string, tag)
    
    elif source == 'ceda':

        return ceda_era_fp(datetime_str, type_string)
    
    else:

        raise Exception('Invalid specification of source for ERA filepath')


def round_to_hours(t):
    # Rounds datetime obj to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +dt.timedelta(hours=t.minute//30))


def get_files(dir, match_every, match_one):
    """
    locates files at dir, whose filenames contain every entry in list match_every and at least one of match_one
    example: match_every is an instrument name, match_one is a batch of dates to select
    """
    # catches case where only one string given
    if isinstance(match_every, str):
        match_every = [match_every]
    
    files = [dir + file for file in os.listdir(dir) if all([x in file for x in match_every]) and any([y in file for y in match_one])]
    return files


def epoch_time_conversion(time_values, epoch=[2000,1,1,0,0,0], use_numpy_dates=True):
    """
    Takes time values in days since epoch and produces datetime or numpy datetime objects 
    """

    if use_numpy_dates:
        data_type = np.datetime64
    else:
        data_type = dt.datetime

    times = np.array([dt.datetime(*epoch) + dt.timedelta(days=time) for time in time_values], dtype=data_type)

    return times


def byte_timestamp_conversion(time_values, use_numpy_dates=True):

    if use_numpy_dates:
        data_type = np.datetime64
    else:
        data_type = dt.datetime

    times = np.array([dt.datetime.strptime(date_time.decode(), "%Y%m%dT%H%M%SZ") for date_time in time_values], dtype=data_type)

    return times



def wrap_str(array, write_fmt, n_cols, row1_ncols=None):
    """
    Format entries of array with the format specified by write_fmt, and output as a string
    with n_cols columns. The first column can be set to have row1_ncols columns independently.
    """

    n_entries = len(array)

    if not row1_ncols:
        row1_ncols = n_cols

    if n_entries <= row1_ncols:
        n_rows_tomake = 1
    else:
        n_rows_tomake = 1 + int( np.ceil( (n_entries - row1_ncols) / n_cols) )

    n_cols_per_row = [n_cols] * n_rows_tomake
    n_cols_per_row[0] = row1_ncols

    row_entry_bounds = np.cumsum([0]+n_cols_per_row)

    output_string = ''

    for i in range(n_rows_tomake):
        output_string += ''.join(['{:{fmt}}'.format(entry, fmt=write_fmt)
                                  for entry in array[row_entry_bounds[i]:row_entry_bounds[i+1]]])
        output_string += '\n'

    return output_string


def round_to_res(x, res=0.25):
    divisor = round(1/res)

    rounded_x = np.round(x * divisor) / divisor

    return rounded_x