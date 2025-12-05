#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
earthquake_fns.py

"""

import numpy as np
import pandas as pd
from datetime import datetime

def get_coastlines(coasts_file):
    """
    Read coastline longitudes and latitudes from a .csv file.
    """
    try:
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

    for plate_name in df["name"].unique():
        subset = df[df["name"] == plate_name]
        coords = np.column_stack(
            (subset["lon"].to_numpy(), subset["lat"].to_numpy())
        )
        pb_dict[plate_name] = coords

    return pb_dict


def get_earthquakes(filename):
    """
    Read an earthquake .csv file into a pandas DataFrame.
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
    """
    lats = df["Latitude"].to_numpy()
    lons = df["Longitude"].to_numpy()
    depths = df["Depth"].to_numpy()
    magnitudes = df["Magnitude"].to_numpy()

    time_series = pd.to_datetime(df["Time"], format='mixed')
    times = time_series.dt.to_pydatetime()

    return lats, lons, depths, magnitudes, times


def select_quake_subset(df, times=None, lons=None, lats=None, depths=None, mags=None):
    """
    Extract a subset of an earthquake DataFrame based on specific criteria.
    Returns the original dataframe if no optional arguments are provided.
    """
    df_sub = df.copy()

    # Convert Time column to datetime to ensure comparisons work correctly
    if times is not None:
        df_sub['Time'] = pd.to_datetime(df_sub['Time'], format='mixed')
        # Handle cases where input times might not be pandas timestamps
        t_start = pd.to_datetime(times[0])
        t_end = pd.to_datetime(times[1])
        df_sub = df_sub[(df_sub['Time'] >= t_start) & (df_sub['Time'] <= t_end)]

    if lons is not None:
        df_sub = df_sub[(df_sub['Longitude'] >= lons[0]) & (df_sub['Longitude'] <= lons[1])]

    if lats is not None:
        df_sub = df_sub[(df_sub['Latitude'] >= lats[0]) & (df_sub['Latitude'] <= lats[1])]
        
    if depths is not None:
        df_sub = df_sub[(df_sub['Depth'] >= depths[0]) & (df_sub['Depth'] <= depths[1])]

    if mags is not None:
        df_sub = df_sub[(df_sub['Magnitude'] >= mags[0]) & (df_sub['Magnitude'] <= mags[1])]

    return df_sub


def get_slope(start_pt, end_pt):
    """
    Calculate the slope of a line in degrees between two (x, y) points.
    
    Parameters
    ----------
    start_pt : tuple or list
        (x1, y1) coordinates in km
    end_pt : tuple or list
        (x2, y2) coordinates in km
        
    Returns
    -------
    slope_deg : float
        Slope angle in degrees
    """
    x1, y1 = start_pt
    x2, y2 = end_pt
    
    rise = y2 - y1
    run = x2 - x1
    
    if run == 0:
        raise ValueError("Division by zero: The x-coordinates of the start and end points cannot be the same.")
        
    slope = rise / run
    
    # Return angle in degrees
    return np.degrees(np.arctan(slope))


# In[ ]:




