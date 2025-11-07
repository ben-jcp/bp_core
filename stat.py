"""
Functions for calculating statistics.

Author: Ben Pery
"""
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid

def calculate_mean(f_x, x):
    """
    Calculates the mean of a 1-dimensional density distribution.

    Arguments:
        :f_x:
        :x:

    Returns:
        :expectation_x:
    """

    expectation_x = np.divide(trapezoid(f_x * x, x), trapezoid(f_x, x))
    return expectation_x


def get_median(f_x, x, pctl=0.5):
    cumul_response = cumulative_trapezoid(f_x, x)
    pctl_loc = np.argmin(np.abs((cumul_response/np.max(cumul_response)) - pctl))
    pctl_spec = x[pctl_loc]

    return pctl_spec


def get_stdev(f_x, x,):
    expectation_x = np.divide(trapezoid(f_x * x, x), trapezoid(f_x, x))
    expectation_x_squared = np.divide(trapezoid(f_x * (x**2), x), trapezoid(f_x, x))
    stdev = np.sqrt(expectation_x_squared - expectation_x**2)

    return stdev


def uncertainties_avg(data, err_data, return_meas_avg=False):

    # measurement uncertainty term: sum^N (err^2) / N
    meas_err = np.nanmean(err_data ** 2,axis=0)

    # averaging uncertainty term: (sample stdev)^2 / N
    avging_error = np.nanstd(data, axis=0, ddof=1) ** 2 / np.sum(np.logical_not(np.isnan(data)),axis=0) 
        # denominator ensures each part of 2D array averaged correctly

    total_error = np.sqrt(meas_err + avging_error)
    if return_meas_avg:
        return total_error, np.sqrt(meas_err), np.sqrt(avging_error)
    else:
        return total_error