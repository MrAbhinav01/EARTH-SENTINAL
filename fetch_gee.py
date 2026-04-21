

import ee
import os
import subprocess
from datetime import datetime, timedelta


ee.Initialize(project='satellite-472011')


states = ee.FeatureCollection("FAO/GAUL/2015/level1")
hp = states.filter(ee.Filter.eq('ADM1_NAME', 'Himachal Pradesh'))
himachal = hp.geometry()


def getS1VVVH(start, end):
    vv = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(himachal) \
        .filterDate(start, end) \
        .filter(ee.Filter.eq('instrumentMode','IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')) \
        .select('VV')

    vh = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(himachal) \
        .filterDate(start, end) \
        .filter(ee.Filter.eq('instrumentMode','IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')) \
        .select('VH')

    vvImg = ee.Image(ee.Algorithms.If(vv.size().gt(0), vv.median().clip(himachal), ee.Image(0).rename('VV')))
    vhImg = ee.Image(ee.Algorithms.If(vh.size().gt(0), vh.median().clip(himachal), ee.Image(0).rename('VH')))

    return ee.Image.cat([
        vvImg.multiply(1000).toInt16(),
        vhImg.multiply(1000).toInt16()
    ]).resample('bilinear')


def getChirps(start, end):
    return ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterBounds(himachal) \
        .filterDate(start, end) \
        .sum() \
        .clip(himachal) \
        .multiply(10).toInt16() \
        .resample('bilinear') \
        .rename('rainfall')

def getSlope():
    dem = ee.Image('USGS/SRTMGL1_003').clip(himachal)
    return ee.Terrain.slope(dem).multiply(100).toInt16().resample('bilinear').rename('slope')

slope = getSlope()

def stackWeek(start, end):
    return ee.Image.cat([
        getS1VVVH(start, end),
        getChirps(start, end),
        slope
    ]).rename(['VV','VH','rainfall','slope'])

# -------------------------
# 4) Generate weekly dates (example: monsoon season 14 weeks)
# -------------------------
start_dates = [
  '2016-06-01','2016-06-08','2016-06-15','2016-06-22',
  '2016-06-29','2016-07-06','2016-07-13','2016-07-20',
  '2016-07-27','2016-08-03','2016-08-10','2016-08-17',
  '2016-08-24','2016-08-31'
]




# Download weekly GeoTIFFs  and save to drive 

for i, s in enumerate(start_dates):
    start = ee.Date(s)
    end = start.advance(7, 'day')

    img = stackWeek(start, end)

    print(f"[🚀] Exporting Week-{i+1} to Google Drive...")

    task = ee.batch.Export.image.toDrive(
        image=img.reproject('EPSG:4326', None, 20),
        description=f"HP_week_{i+1}",
        folder="HP_SAT_EXPORTS",              # <-- this folder will appear in your Google Drive
        fileNamePrefix=f"HP_week_{i+1}",
        scale=20,
        region=himachal,
        maxPixels=1e13
    )
    task.start()



print("✅ All exports started.")