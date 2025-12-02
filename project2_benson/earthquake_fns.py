#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
earthquake_fns.py

Core functions for Project Lab 2, Part 1.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def get_coastlines(coasts_file):
    """
    Read coastline longitudes and latitudes from a .csv file.

    Parameters
    ----------
    coasts_file : str
        Path to a .csv file containing two columns:
        column 1 = longitudes of the world's coastlines (degrees),
        column 2 = latitudes of the world's coastlines (degrees).

    Returns
    -------
    lon_coast : numpy.ndarray
        1D array of coastline longitudes in degrees.
    lat_coast : numpy.ndarray
        1D array of coastline latitudes in degrees.

    Raises
    ------
    IOError
        If the file cannot be opened or read.
    """
    try:
        # The file does not need column names, so we add our own.
        df = pd.read_csv(coasts_file, header=None, names=["lon", "lat"])
    except (OSError, FileNotFoundError) as err:
        raise IOError(f"Could not read coastline file '{coasts_file}': {err}")

    lon_coast = df["lon"].to_numpy()
    lat_coast = df["lat"].to_numpy()

    return lon_coast, lat_coast


def get_plate_boundaries(plates_file):
    """
    Read plate-boundary coordinates from a .csv file and
    return them as a dictionary.

    Parameters
    ----------
    plates_file : str
        Path to a .csv file containing three columns:
        column 1 = plate boundary name (abbreviation, str),
        column 2 = latitude in degrees (float),
        column 3 = longitude in degrees (float).

    Returns
    -------
    pb_dict : dict
        Dictionary with one key–value pair for each unique plate name.
        Keys are plate name abbreviations (str).
        Values are (N x 2) numpy arrays of floats with:
            column 0 = longitudes in degrees,
            column 1 = latitudes in degrees.

    Raises
    ------
    IOError
        If the file cannot be opened or read.
    """
    try:
        df = pd.read_csv(
            plates_file,
            header=None,
            names=["name", "lat", "lon"]
        )
    except (OSError, FileNotFoundError) as err:
        raise IOError(f"Could not read plate-boundary file '{plates_file}': {err}")

    pb_dict = {}

    # Loop over each unique plate name and build an array for that plate.
    for plate_name in df["name"].unique():
        subset = df[df["name"] == plate_name]
        # N x 2 array: [lon, lat]
        coords = np.column_stack(
            (subset["lon"].to_numpy(), subset["lat"].to_numpy())
        )
        pb_dict[plate_name] = coords

    return pb_dict


def get_earthquakes(filename):
    """
    Read an earthquake .csv file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to a .csv file. In this project this will be the
        global earthquake catalogue from IRIS.

    Returns
    -------
    earthquakes : pandas.DataFrame
        DataFrame containing the contents of the .csv file.

    Raises
    ------
    IOError
        If the file cannot be opened or read.
    """
    try:
        earthquakes = pd.read_csv(filename)
    except (OSError, FileNotFoundError) as err:
        raise IOError(f"Could not read earthquake file '{filename}': {err}")

    return earthquakes


def parse_earthquakes_to_np(df):
    """
    Extract columns from a DataFrame of earthquake data into
    separate 1D numpy arrays and datetime objects.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing earthquake data. It must contain
        at least the columns:
        'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Time'.

    Returns
    -------
    lats : numpy.ndarray
        1D array of earthquake latitudes in degrees.
    lons : numpy.ndarray
        1D array of earthquake longitudes in degrees.
    depths : numpy.ndarray
        1D array of earthquake depths in kilometres.
    magnitudes : numpy.ndarray
        1D array of earthquake magnitudes (unitless).
    times : numpy.ndarray
        1D array of datetime.datetime objects representing
        earthquake origin times.

    Notes
    -----
    The 'Time' column in the input DataFrame is assumed to be a
    string in ISO format, e.g. '2000-10-31T01:30:00.000-05:00'.
    """
    # Extract numeric data as numpy arrays
    lats = df["Latitude"].to_numpy()
    lons = df["Longitude"].to_numpy()
    depths = df["Depth"].to_numpy()
    magnitudes = df["Magnitude"].to_numpy()

    # Convert Time column from string to datetime objects
    # First convert to pandas datetime, then to Python datetime
    time_series = pd.to_datetime(df["Time"], format = 'mixed')
    times = time_series.dt.to_pydatetime()

    return lats, lons, depths, magnitudes, times


# In[ ]:




