import numpy as np


def pixel_centres_in_circle_by_xtr(sat_geo_ds, centre_coords, radius_deg, xtrack_range=None):

    ref_lat, ref_lon = centre_coords

    pix_in_range_by_xtr = {}

    if not xtrack_range:
        xtrack_range = sat_geo_ds['xtrack'].size

    for xtr in xtrack_range:

        lats = sat_geo_ds['latitude'][:,xtr].values
        lons = sat_geo_ds['longitude'][:,xtr].values

        coord_dists = np.sqrt( (lats-ref_lat)**2 + (lons-ref_lon)**2 )
        in_range_pixels = np.nonzero(coord_dists < radius_deg)[0]

        # in_range_pixels = []
        # for atr, (lat, lon) in enumerate(zip(sat_geo_ds['latitude'][:,xtr], sat_geo_ds['longitude'][:,xtr])):
        #     coord_dist = np.sqrt( (lat-ref_lat)**2 + (lon-ref_lon)**2 )

        #     if coord_dist < radius_deg:
        #         in_range_pixels += [atr]

        n_in_range = len(in_range_pixels)
        if n_in_range > 0:
            pix_in_range_by_xtr[xtr] = in_range_pixels
            print(f'{n_in_range} pixels found for x-track sensor {xtr+1}')
        else:
            print(f'No pixels found for x-track sensor {xtr+1}')

    return pix_in_range_by_xtr



def calc_degrees_goes_abi(goes_ds):
# Calculate latitude and longitude from GOES ABI fixed grid projection data
# GOES ABI fixed grid projection is a map projection relative to the GOES satellite
# Units: latitude in °N (°S < 0), longitude in °E (°W < 0)
# See GOES-R Product User Guide (PUG) Volume 5 (L2 products) Section 4.2.8 for details & example of calculations
# "goes_ds" is an ABI L1b or L2 .nc file opened using the xarray library
# slightly modified by BP 11 Nov 2025 from https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = goes_ds['x'].values  # E/W scanning angle in radians
    y_coordinate_1d = goes_ds['y'].values  # N/S elevation angle in radians
    projection_info = goes_ds['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon