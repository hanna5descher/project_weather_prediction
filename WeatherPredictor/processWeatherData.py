"""
This function processes downloaded NOAA zip files into a single archive of hourly data
"""

import os
import zipfile
import pandas as pd
import datetime as dt


weatherDir = "weatherData"

print(os.getcwd())
# Extract any data not already extracted
for file in os.listdir(weatherDir):
    if file.endswith('.zip'):
        curFile = os.path.join(weatherDir, file)
        with zipfile.ZipFile(curFile, 'r') as cZip:
            for member in cZip.namelist():
                destination = os.path.join(weatherDir, member)
                if not os.path.exists(destination):
                    print('Unzipping File: ', member)
                    cZip.extract(member, weatherDir)

# Hourly Weather Data
if False:
    # Collate all hourly data
    hourlyDataFileName = 'hourlyData.hdf'
    if os.path.isfile(hourlyDataFileName):
        os.remove(hourlyDataFileName)

     
    for file in os.listdir(weatherDir):
        if file.endswith("hourly.txt"):
            print('Processing file: ', file)
            df = pd.read_csv(os.path.join(weatherDir,file),
                            index_col=['WBAN', 'Date', 'Time'],
                             na_values=('M', '  ', ' ', 'VR', 'VR ',  "  T", "   ", 'null'),
                             usecols=['WBAN', 'Date', 'Time', 'DryBulbFarenheit', 'DewPointFarenheit', 'RelativeHumidity',
                                      'WindSpeed', 'WindDirection', 'StationPressure', 'HourlyPrecip'],
                             parse_dates=True
                             )
            df = df.apply(pd.to_numeric, errors='coerce')
            if not os.path.isfile(hourlyDataFileName):
                df.to_hdf(hourlyDataFileName, 'hourlyData', format='t', mode='w', complevel=1, complib='zlib')
            else:
                df.to_hdf(hourlyDataFileName, 'hourlyData', format='t', mode='a', append=True, complevel=1, complib='zlib')

# Station Data
if True:
    print('Getting Station Info')
    cols = None
    # Collate all hourly data
    stationDataFileName = 'stationData.hdf'
    if os.path.isfile(stationDataFileName):
        os.remove(stationDataFileName)
     
    for file in os.listdir(weatherDir):
        if file.endswith('station.txt'):
            print('Processing file: ', file)
            if cols is None:
                df = pd.read_csv(os.path.join(weatherDir,file),
                                 sep='|',
                                 parse_dates=True)                
            else:
                pd.read_csv(os.path.join(weatherDir,file),
                                 sep='|',
                                 parse_dates=True,
                                 usecols=cols)
            if not os.path.isfile(stationDataFileName):
                df.to_hdf(stationDataFileName, 'stationInfo', format='t', mode='w', complevel=1, complib='zlib')
                cols = list(df)
            else:
                df.to_hdf(stationDataFileName, 'stationInfo', format='t', mode= 'a', append=True, complevel=1, complib='zlib')

# Reprocess Hourly Weather Data
dirWeather = "dataWeather"
dirStation = "dataStation"
dfStation = pd.read_csv(os.path.join(dirStation,"dataStation.csv"));
print(dfStation.head())

dfs=[]
index=0
for station in dfStation.WBAN:
    index+=1
    print('Getting Data for Station', index, 'of', len(dfStation.WBAN))
    where = 'WBAN==' + str(station)
    dataTempPerStation = pd.read_hdf(os.path.join(dirWeather,'hourlyData.hdf'),where=where)
    dataDateTime = dataTempPerStation.index.values
    for t in range(0,len(temp)):

    dfs.append(temp)
    if index > 3:
         break

