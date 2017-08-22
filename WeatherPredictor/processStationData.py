"""
This script processes downloaded station data from NOAA and performs K-means clustering based on locations
"""

import os
import pandas as pd
import numpy as np
import scipy
from sklearn import cluster
import matplotlib.pyplot as plt

dirStation = "dataStation"
dateStation = [200705,201704] # define station files to load
nameState = ['NC','VA','TN','GA','SC'] # define NC and neiboring states
coordHome = [35.923717, -78.946995] # home coordinate from Google Maps
nameVar = ['WBAN','State','Name','Latitude','Longitude'] # variables of interest

# Organize station data
dfAllStation = {}
for d in range(len(dateStation)):
    file = str(dateStation[d]) + 'station.txt'
    print('Processing file: ', file)
    dfAllStation[d] = pd.read_csv(os.path.join(dirStation,file),
                        sep='|',
                        parse_dates=True) 

dfStation = pd.DataFrame()
for state in nameState:
    print('Processing state: ', state)
    tempState1 = dfAllStation[0].loc[dfAllStation[0]['State']==state,nameVar]
    tempState2 = dfAllStation[1].loc[dfAllStation[1]['State']==state,nameVar]
    tempState = pd.merge(tempState1, tempState2, how='inner', on= ['WBAN'])
    tempState = tempState.iloc[:,:len(nameVar)] # for some reason, coordinate values are slightly different
    tempState.columns = nameVar
    dfStation = dfStation.append(tempState)        

dfStation = dfStation.set_index(['WBAN'])
distFromHome = np.sqrt((coordHome[0]-dfStation['Latitude'])**2 + (coordHome[1]-dfStation['Longitude'])**2)
homeStation = dfStation.loc[distFromHome == min(distFromHome),:] # determine the closest weather station
dfStation['relLatitude'] = dfStation['Latitude']-homeStation['Latitude'].values[0]
dfStation['relLongitude'] = dfStation['Longitude']-homeStation['Longitude'].values[0]

# Run K-means clustering of weather stations based on their locations
# also, try using DBSCAN
k = 10
dataKMeans = dfStation[['relLatitude','relLongitude']].values
kmeans = cluster.KMeans(n_clusters=k, n_init=100)
kmeans.fit(dataKMeans)
dfStation['cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_
dfStationCentroid = dfStation.loc[[homeStation.index.values[0]]] # set first row as the home station
for i in range(k):    
    ds = dataKMeans[np.where(kmeans.labels_==i)] # select only data observations with cluster label == i

    # Find weather stations closest to the centroids
    distCentroids = np.sqrt((centroids[i,0]-dataKMeans[:,0])**2 + (centroids[i,1]-dataKMeans[:,1])**2)
    dfStationCentroid = dfStationCentroid.append(dfStation.loc[distCentroids == min(distCentroids),:]) # determine the closest weather station

    # Plot results
    plt.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
plt.axhline(0)
plt.axvline(0)
plt.show()

# save data
dfStation.to_csv(os.path.join(dirStation,'dataStationOfInterest.csv'))
dfStationCentroid.to_csv(os.path.join(dirStation,'dataStationCentroids.csv'))