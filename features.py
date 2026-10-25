import numpy as np
import pandas as pd
TARGET = 'Rented Bike Count'
RENAME = {'Temperature(°C)': 'Temperature', 'Humidity(%)': 'Humidity', 'Wind speed (m/s)': 'Wind speed', 'Visibility (10m)': 'Visibility', 'Dew point temperature(°C)': 'Dew point temperature', 'Solar Radiation (MJ/m2)': 'Solar Radiation', 'Rainfall(mm)': 'Rainfall', 'Snowfall (cm)': 'Snowfall'}

def prepare_features(raw):

    frame = raw.rename(columns=RENAME).copy()
    dates = pd.to_datetime(frame.pop("Date"), format="%d/%m/%Y", errors="raise")
    frame = frame.drop(columns=[TARGET], errors="ignore")
    frame["Month"] = dates.dt.month
    frame["Day of week"] = dates.dt.dayofweek
    frame["Day of year"] = dates.dt.dayofyear
    frame["Weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    frame["Hour sin"] = np.sin(2 * np.pi * frame["Hour"] / 24)
    frame["Hour cos"] = np.cos(2 * np.pi * frame["Hour"] / 24)
    frame["Month sin"] = np.sin(2 * np.pi * (frame["Month"] - 1) / 12)
    frame["Month cos"] = np.cos(2 * np.pi * (frame["Month"] - 1) / 12)
    return frame
