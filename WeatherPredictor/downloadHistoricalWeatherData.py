"""
This file downloads all NOAA Local Climate Data to weatherData subfolder

Data Location:
https://www.ncdc.noaa.gov/orders/qclcd/

Only data after 2007 05 is used, due to unit change before that date.

"""

import urllib.request
from dateutil.relativedelta import relativedelta
import datetime
import os

RemotePath = "https://www.ncdc.noaa.gov/orders/qclcd/"
curDate = datetime.datetime(2007, 5, 1)
stopDate = datetime.datetime(2017, 4, 1)

weatherDir = "weatherData"
if not os.path.exists(weatherDir):
    os.makedirs(weatherDir)

while curDate <= stopDate:
    curFile = curDate.strftime("QCLCD%Y%m.zip")
    remoteFile = urllib.request.urlopen(RemotePath + curFile)
    curFile = os.path.join(weatherDir, curFile)

    if not remoteFile.isclosed():
        if not (os.path.isfile(curFile) and os.stat(curFile).st_size == remoteFile.length):
            print('Downloading file: ' + curFile)

            fileData = remoteFile.read()
            remoteFile.close()

            localFile = open(curFile, mode='wb')
            localFile.write(fileData)
            localFile.close()

    curDate += relativedelta(months=1)

