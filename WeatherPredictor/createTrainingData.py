"""
This script cleans up and re-formats weather data appropriate for tensorflow
"""
import os
import pandas as pd
import numpy as np
import datetime

"""
Define Functions
"""
def getWeatherStationData(station,dirWeatherData): # load weather data per station
    where = 'WBAN==' + str(station)
    df = pd.read_hdf(dirWeatherData, where=where) 
    return df

def switchIndex(df): # re-index weather data
    # load index
    wban =  df.index.get_level_values(0)
    dates = df.index.get_level_values(1)
    times = df.index.get_level_values(2)    

    # merge date and time indices (correct timestamp)
    times = times.astype(np.int32)
    indexDate = []
    index = 0
    for date in dates:
        time = times[index].item(0)
        dt = datetime.timedelta(hours = time//100, minutes = time%100) 
        indexDate.append(date+dt) 
        index+=1
    df.index = indexDate # reindex data (merge to a single timestamp index)

    # rename columns to include WBAN (unique station ID)
    columns = df.columns.values
    for i, column in enumerate(columns):
        columns[i]=columns[i] + '_' + str(wban[0])
    df.columns = columns

    return df[~df.index.duplicated(keep='first')]

"""
Main Script
"""
# load stations in NC + neighboring states (see processStationData.py)
dirWeather = 'dataWeather'
dirStation = 'dataStation'
nameStationFile = 'dataStationOfInterest.csv' 
dataStation = pd.read_csv(os.path.join(dirStation,nameStationFile))

# reindex weather data (some weather stations contain inconsistent timestamps and/or number of data points)
dirWeatherData = os.path.join(dirWeather,'hourlyData.hdf')
dfWeatherSOI_allTime = []
timeDataStart = []
timeDataEnd = []
for index, station in enumerate(dataStation['WBAN']):
    print('processing station wban', station, ', index', index, 'of',len(dataStation))

    df = getWeatherStationData(station,dirWeatherData)

    if len(df) > 0:
        df = switchIndex(df) # re-index weather data

        # filter out stations that have lots of missing data
        # include station if its records start before 2009 and extend beyond 1/1/2017
        if df.index[0] < datetime.datetime(2009,1,1) and df.index[-1] > datetime.datetime(2017,1,1):            
            dfWeatherSOI_allTime.append(df)
            timeDataStart.append(df.index[0])
            timeDataEnd.append(df.index[-1])
    else:
        print('unable to load data from station', station, ', index', index)

# create new hourly index to have uniform timestamps across all stations 
timeDataStartMax = max(timeDataStart).replace(minute=0) # get the latest start time (consider using + datetime.timedelta(hours=1))
timeDataEndMax = max(timeDataEnd).replace(minute=0) # get the latest end time (there's not much variation) (consider using - datetime.timedelta(hours=1))
indexRange = pd.date_range(timeDataStartMax,timeDataEndMax,freq='H')

# reindex and merge data by hour
for numdf, df in enumerate(dfWeatherSOI_allTime):
    print('Reindexing', numdf, 'of', len(dfWeatherSOI_allTime))   
    dfWeatherSOI_allTime[numdf] = df.reindex(indexRange,method='nearest') 

# concatenate and save data
print('Concatenating', len(dfWeatherSOI_allTime), 'station data, from',timeDataStartMax, 'to', timeDataEndMax)
dfWeatherSOI = pd.concat(dfWeatherSOI_allTime,axis=1)
dfWeatherSOI.to_hdf(os.path.join(dirWeather,'dataTrainingWeatherSOI.hdf'), 'dataTrainingWeatherSOI', format='t', mode='w', complevel=1, complib='zlib')


        