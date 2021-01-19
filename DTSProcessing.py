#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NamTEX
DTS Functions for actual use

Upgrading the old DTSHelperFunctions.py now that I have a better handle on how to treat
this data. Also incorporating the data storage approach of the dtscalibration package.
Testing should take place in DTSHelperFunctions.py or Calibration_Testing.py

This script will incorporate methods that work from those to produce useful output

Created on Wed Sep 16 14:54:46 2020

@author: RainerHilland
"""

# packages
from datetime import datetime, timedelta
import os, glob, scipy
import dtscalibration as dts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import deepdish as dd

#%% Organisation and Calibration Functions

def getTime(fName):
    
    '''
    Strips the time from silixa file names
    
    Format: channel_[n]_[tz]_[YYYYMMDD]_[HHMMSS.SSS].xml -> time is END of sample!
    
    Input: silixa file name
    Output: datetime object timestamp of the end of the sample
    '''
    
    fName = os.path.splitext(os.path.basename(fName))[0] # strip the file path stuff
    ymd, hms = fName.split('_')[2:]
    
    timeStamp = datetime.strptime(ymd+hms, '%Y%m%d%H%M%S.%f')
    
    return(timeStamp)


def defineChunks(targetTimes, fileNames, fileTimes=None):
    
    '''
    Defines the individual chunks for processing raw DTS files
     ** 30-MINUTE CHUNKS HARDCODED at the mo **
     
    Parameters
    -----------
    targetTimes: a list of times that define the starts of chunks for processing
    fileNames: a (sorted) list of all the file names for consideration
    fileTimes: optional, a list of extracted times from those filenames.
    
    Returns
    --------
    chunks: a list of tuples with information needed to process the chunks of files
    '''
    
    if fileTimes is None:
        # start by stripping the times from the filenames if not provided
        fileTimes = [getTime(file) for file in fileNames]
    
    # collect the nearest (greater) file data
    # chunkSplits is a list of tuples with organisation:
        # (target time, closest time, closest time index, closest file name)
    chunkSplits = [findNearest(t, fileTimes, fileNames) for t in targetTimes]
    
    chunks = []
    for idx, entry in enumerate(chunkSplits):
        
        if idx == len(chunkSplits)-1: continue
            
        chunkStartTime = chunkSplits[idx][0]
        chunkEndTime = chunkStartTime + timedelta(seconds=1799) # should find a way to reflect underlying data actually
        chunkStartIndex = chunkSplits[idx][2]
        chunkEndIndex = chunkSplits[idx+1][2]-1
        chunkStartName = chunkSplits[idx][3]
        chunkEndName = fileNames[chunkEndIndex]
        
        if entry[1] - entry[0] > timedelta(seconds=1800):
            # closest time to the start time is > 30 minutes away;
            # this start time has no files
            chunkStartIndex = None
        if fileTimes[chunkEndIndex] - chunkEndTime > timedelta(seconds=1800):
            # end times mismatch by > 30 minutes, kill the chunk
            chunkStartIndex = None

        
        chunks.append((chunkStartTime, chunkEndTime, chunkStartIndex,
                       chunkEndIndex, chunkStartName, chunkEndName))
        
    return(chunks)
    
def findNearest(target, times, fileNames):
    
    '''
    Finds the nearest time (over) a target time from a series of input times.
    
    Parameters
    -----------
    target: the time you want to calculate distance from
    times: a list of all the file times you have 
    fileNames: a list of the file names
    
    =-> times and fileNames should be sorted such that they're 1:1
    
    Returns
    --------
    target: the reference/target time
    closestTime: the closest time within param:times to the target that is > target
    closestTimeIndex: where in the list of files/names/etc. this is
    closestFile: the file at index param:closestTimeIndex
    
    '''
    
    closestTime = min(times, key=lambda x: searchFunction(x, target))
    closestTimeIndex = times.index(closestTime)
    closestFile = fileNames[closestTimeIndex]
    
    return(target, closestTime, closestTimeIndex, closestFile)
    
    
def searchFunction(x, target):
    
    '''
    Finds the distance between an input time (x) and a target time (target)
    '''
    
    delta = abs(x-target) if x > target else timedelta.max # only take times AFTER
    return delta


def getTimeSequence(date, interval=30):
    
    '''
    This function produces the time sequence list used to divide up the DTS days.
    
    Parameters
    -----------
    date: the day for which the sequence is to be produced. format 'DD.MM'
    interval: how many minutes per interval
    
    Returns: the sequence
    '''
    
    day, month = date.split('.')
    startString = '2020'+month+day+'000000'
    endString = '2020'+month+str(int(day)+1).zfill(2)+'003000'
    
    start = datetime.strptime(startString, '%Y%m%d%H%M%S')
    end = datetime.strptime(endString, '%Y%m%d%H%M%S')
    
    seconds = (end-start).total_seconds()
    step = timedelta(minutes=interval)
    
    seq = []
    for i in range(0, int(seconds), int(step.total_seconds())):
        seq.append(start + timedelta(seconds=i))
        
    return(seq)
    
    
def readChunk(fileNames):

    ''' 
    Reads a chunk and sets the calibration sections
    
    Input: the filenames to read (hopefully already sorted!)
    Returns: the datastore object
    '''
    
    ds = dts.read_silixa_files(filepathlist=fileNames,
                timezone_netcdf='UTC', file_ext='*.xml')
    
    # these never change   
    # still something strange with second cold bath and the variance signal...
    sections = {
    'probe1Temperature': [slice(59., 167.), slice(2659., 2746.)], # p1 = cold bath
    'probe2Temperature': [slice(171., 247.), slice(2558., 2656.)] # p2 = warm bath
    }
    ds.sections = sections
    
    return(ds)
    
    
def calibrateChunk(ds, calculateIntervals=True):
    
    ''' 
    Performs the calibration of a chunk.
    
    Parameters
    -----------
    ds: a DTS DataStore object with more than one timestep
    calculateIntervals: T/F calculate confidence intervals?
    
    Returns the stokes variance and residuals from this chunk.
    '''
    
    # stokes and anti-stokes variance
    # using a constant correction - prefer to use linear but that function
    # seems to be broken / perhaps I can figure it out?
    print("variances")
    stokesVariance, stokesResiduals = ds.variance_stokes_constant(st_label='st')
    antiStokesVariance, _ = ds.variance_stokes_constant(st_label='ast')
    
    print("T cal")
    # perform the calibration
    ds.calibration_single_ended(sections=ds.sections, st_var=stokesVariance, 
                                ast_var=antiStokesVariance, method='wls')
    
    # calculate the confidence intervals?
    # this is TIME CONSUMING! and I'm not sure it makes sense over a group?
    if calculateIntervals:
        print("intervals")
        ds.conf_int_single_ended(st_var=stokesVariance, ast_var=antiStokesVariance,
                                 conf_ints=[2.5, 97.5], mc_sample_size=500)
        
    # ds methods should modify the object directly, so I won't return them. But check that.
    return(stokesVariance, stokesResiduals)


def saveCalibratedFile(ds, folder):
    
    ''' saves a calibrated file... '''
    
    # write the timing information out just as fyi
    timingFileName = os.path.join(folder, 'timing.txt')
    tFile = open(timingFileName, 'w')
    tFile.write(str(ds.isel(time=0).time.data)[:-5]+'\n')
    tFile.write(str(ds.isel(time=-1).time.data)[:-5])
    tFile.close()
    
    # save the file as an .nc
    # name is hardcoded at the mo
    ncFileName = os.path.join(folder, 'calibratedDTS_DATA.nc')
    ds.to_netcdf(ncFileName)
    
    return(True)


def helpfulPlots(ds, resid, folder, st_var):
    
    ''' 
    Produces some helpful plots!
    
    Parameters
    ------------
    ds: a datastore object, single timestep, with .tmpf variable
    resid: the residuals object from the calibration
    folder: the folder in which to write the plots out
    st_var: the stokes variance from the calibration
    
    Returns nothing, just writes plots.
    '''
    
    plotTiming = str(ds.isel(time=0).time.data)[:-7] + ' to ' + str(ds.isel(time=-1).time.data)[:-7]

    
    # calibration sections (the mystery plot, tbh)
    fig = plt.figure(figsize=(24,18))
    fig2 = dts.plot.plot_residuals_reference_sections(resid, ds.sections, plot_avg_std=st_var**0.5,
                                                      plot_names=True, robust=True, method='split',
                                                      fig=fig)
    plt.title(plotTiming)
    fig2.savefig(os.path.join(folder, 'referenceResiduals.png'))
    plt.close(fig)
    plt.close(fig2)
    
    
    # residual distributions
    sigma = resid.std()
    mean = resid.mean()
    x = np.linspace(mean-3*sigma, mean+3*sigma, 100)
    approxNormFit = scipy.stats.norm.pdf(x, mean, sigma)
    fig,ax=plt.subplots(figsize=(18,12))
    resid.plot.hist(bins=50, density=True, ax=ax)
    ax.plot(x, approxNormFit)
    plt.suptitle("Stokes Residual Distribution")
    plt.title(plotTiming)
    plt.savefig(os.path.join(folder, 'stokesResidDist.png'))
    plt.close(fig)
    
    # plot the full time series in a color-mappy kind of way
    fig,ax = plt.subplots(figsize=(50,30))
    ds.sel(x=slice(1., 3000.)).tmpf.plot(ax=ax)
    fig.savefig(os.path.join(folder, 'timeSeriesCal.png'))
    plt.title(plotTiming)
    plt.close(fig)
    
    # show avg T and calibration
    dsMean = ds.mean(dim='time', keep_attrs=True)
    fig,ax = plt.subplots(figsize=(24,18))
    dsMean.sel(x=slice(1., 3000.)).tmp.plot(ax=ax, label='uncalibrated')
    dsMean.sel(x=slice(1., 3000.)).tmpf.plot(ax=ax, label='calibrated')
    plt.suptitle('Calibration comparison - mean temp')
    plt.title(plotTiming)
    plt.legend()
    fig.savefig(os.path.join(folder, 'calibrationMean.png'))
    plt.close(fig)
    
    return(True)


def chunkHandler(chunk, files, outMaster):
    
    ''' 
    This function processes the individual chunks determined earlier.
    
    Given a defined chunk, this function reads the individual .nc files,
    calculates variances, calibrates the temperatures, produces some useful diagnostic plots,
    and saves an output DataStore file.
    
    Parameters
    -----------
    chunk: a 6-component tuple defining this section of time
    files: a list of individual .xml file names
    outMaster: the parent folder for output stuff
    
    Returns nothing, but does a lot of processing along the way.
    '''
    
    
    
    startTime, endTime, startI, endI, startName, endName = chunk
    outFolder = os.path.join(outMaster, startTime.strftime('%m.%d_%H%M'))
    if not os.path.isdir(outFolder): os.mkdir(outFolder)
    if os.path.isfile(os.path.join(outFolder, 'calibratedDTS_DATA.nc')):
        print("Already completed this one\n")
        return(False)
    
    # removed 23.10.2020 -> every stop/restart would result in 1 skipped section
    # going to re-do these quickly to remove the false blanks.
    
    t = os.path.join(outFolder, 'note.txt')
    if os.path.isfile(t):
        print("Already completed; cable broken?\n")
        return(False)
        
    
    if startI is None:
        f = open(os.path.join(outFolder, 'note.txt'), 'w')
        f.write("This chunk didn't have enough files\n")
        f.write("Start/End should be:\n")
        f.write(str(startTime)+'\n'+str(endTime)+'\n')
        f.write("Nearest files:\n")
        f.write(startName+'\n'+endName+'\n')
        f.close()
        return(False)
    
    print("reading the chunk")
    
    if endI != -1: # bandaids...
        ds = readChunk(files[startI:endI+1])
    else:
        ds = readChunk(files[startI:endI])
    
    print("calibrating the chunk")
    try:
        st_var, resids = calibrateChunk(ds, calculateIntervals=False)
        
        print("plotting")
        helpfulPlots(ds, resids, outFolder, st_var)
        
        print("saving the chunk")
        saveCalibratedFile(ds, outFolder)
    except:
        print("ERROR w/ calibration - probably broken cable :/")
        f = open(os.path.join(outFolder, 'note.txt'), 'a')
        f.write("Something didn't work during the calibration :(")
        f.close()

    return(True)        
    

#%% Mapping Functions

# these functions are slightly out-of-order but are complementary and stack. If you're
# not doing anything hectic, then for one timestep you should be able to just use 
# fullMapOneTimestep and get the mapped components out. Avging/Plotting available, but
# rudimentary.

# TODO: improve / port the DTSHelperFunctions mapping stuff.
# TODO: automate map selection!

# =-> put things in space in a + instead of an x, will make life easier

def fullMapOneTimestep(mapArray, denseMapArray, ds, component=True, vertMean=True):
    
    '''
    Produces a full map (i.e. entire array, dense sections and all corners)
    
    # TODO: I presume that setting component to false and vertMean to false will break things.
    
    Parameters
    -----------
    mapArray: the map array to use for the long sections
    denseMapArray: the map array to use for the dense sections
    ds: the datastore object with .tmpf variable. Must be one timestep!
    component: T/F return individual components if true, otherwise return tuple of 
                SE->NW and SW->NE
    vertMean: T/F, if true returns are 1-dimensional and averaged in height. The dense
                section in this case will only return the average of the bottom two levels
    
    Returns
    --------
    The map, in a variety of possible formats.
    '''
        
    # TODO: the axes have hardcoded lengths at the moment; should make this more flexible!
    
    NW = longQuarterMapper(mapArray, 'NW', ds=ds, vertMean=vertMean)
    SE = longQuarterMapper(mapArray, 'SE', ds=ds, vertMean=vertMean)
    SW = longQuarterMapper(mapArray, 'SW', ds=ds, vertMean=vertMean)
    NE = longQuarterMapper(mapArray, 'NE', ds=ds, vertMean=vertMean)
    
    NESW, SENW = denseMapper(denseMapArray, ds)
    
    if vertMean:
        NESW = denseAverager(NESW)[:106]
        SENW = denseAverager(SENW)[:106]
    
    if component:
        return(NW,SE,SW,NE,NESW,SENW)
    else:
        if vertMean:
            #SEtoNW = np.empty((SE.shape[0]+SENW.shape[0]+NW.shape[0]), dtype=np.float32)
            #SWtoNE = np.empty((SW.shape[0]+NESW.shape[0]+NE.shape[0]), dtype=np.float32)
            
            SEtoNW = np.append(SE, SENW)
            SEtoNW = np.append(SEtoNW, NW)
            
            SWtoNE = np.append(SW, np.flip(NESW))
            SWtoNE = np.append(SWtoNE, NE)
            
        return(SEtoNW, SWtoNE)
            
            
def singleTraverseMapper(mapArray, counter, ds):
    
    '''
    This maps a single transect, i.e. one height from one corner
    
    Parameters
    ------------
    mapArray: the map array
    counter: where to start in the map array
    ds: the data store object of this timestep
    
    Returns
    --------
    a mapped, fixed, single height transect
    '''
    
    # This modification seems like it should work:
    #print('single traversing')
        
    reverse = int(mapArray['reverse'][counter])
    
    startI = mapArray['idx'][counter]
    endI = mapArray['idx'][counter+1]
    
    section = ds.isel(x=slice(startI,endI+1)).tmpf.data
    indices = np.arange(startI,endI+1)
    
    #d = {'index':indices, 'T':section}
    df = pd.DataFrame.from_dict({'index':indices, 'T':np.array(section)}).set_index('index')
    #df = df.set_index('index')
    
    fixedSection = pointDeleter(mapArray, df, counter+2, counter+15, reverse)
    
    return(fixedSection)


def pointDeleter(mapArray, fibreSegment, start, end, reverse):
    
    '''
    Handles the individual averaging/deleting needed to clean the long transects.
    More info is in that mappingMethods.doc thing I wrote a while ago
    
    Parameters
    ------------
    mapArray: the map array
    fibreSegment: an un-modified transect
    start, end: which part of hte mapArray to use
    reverse: whether or not to flip the cleaned section
    
    Returns
    --------
    a fixed / clean segment.
    '''
    
    #print('point deleting')
    
    for idx in range(start, end+1):
        if mapArray['type'][idx] == 1:
            toDel = mapArray['idx'][idx] # this is the peak to be deleted
            newPt = (fibreSegment.loc[toDel-1] + fibreSegment.loc[toDel+1]) / 2 # avg adjacent points
            
            fibreSegment.loc[toDel-1] = newPt # assign new point
            fibreSegment = fibreSegment.drop([toDel,toDel+1]) # delete points
            
        elif mapArray['type'][idx] == 2:
            toDel = mapArray['idx'][idx]
            fibreSegment = fibreSegment.drop([toDel, toDel+1]) # delete two points
            
    fibreSegment = fibreSegment['T']
    if reverse: fibreSegment = np.flip(fibreSegment)
            
    return(fibreSegment)


def longQuarterMapper(mapArray, direction, ds, vertMean=True):


    ''' 
    Maps a quarter of the array - low section. Either a (2,~600) is returned
    or the observations are averaged vertically and a (1,~600) array is returned
                    
    Parameters
    -----------
    mapArray: a long map array
    direction: string specifying which quarter, NW/SE/SW/NE
    ds: the datastore object of one timestep with .tmpf variable
    vertMean: T/F, should observations be averaged w/ height?
    '''    
    # TODO: implement a patch for the SE section after 10.03 noon
    # =-> depends on map, could check that
    #print('starting a long section')
    
    d = {'NW':0, 'SE':32, 'SW':64, 'NE':96}
    counter = d[direction]
    
    low = singleTraverseMapper(mapArray, counter, ds)
    high = singleTraverseMapper(mapArray, counter+16, ds)
    
    try:
        combi = np.empty((2,high.shape[0]))
        combi[0,:] = high
        combi[1,:] = low
    except:
        difference = high.shape[0] - low.shape[0]
        low = np.flip(low)
        for i in range(difference):
            low = np.append(low, np.nan)
        low = np.flip(low)
        
        combi = np.empty((2,high.shape[0]))
        combi[0,:] = high
        combi[1,:] = low
    
    if vertMean: combi = np.mean(combi, axis=0)
    
    return(combi)


def denseMapper(dMapArray, ds):
    
    '''
    This function performs the dense DataStore -> np.array map routine.
    Mostly ported directly from the DTSHelperFunctions version.
    
    Removed in-function version control; pass a mapArray that's pre-determined instead,
    cuts a lot of unnecessary reading out.
    
    22.09.2020 -> adjusted shape to 108 for V03 map, still returns 107 though,
            last column is mostly empty
    
    Parameters
    ----------
    mapArray: a dense map array telling us how to map
    ds: single timestep DataStore object with .tmpf variable
    
    Returns
    ----------
    2x(16, 108) arrays, one for NESW and one for SENW
    '''
     
    NESW = np.empty((16,108))
    SENW = np.empty((16,108))
    
    for idx in range(2, 18):
        sI = dMapArray['idxStart'][idx]
        eI = dMapArray['idxEnd'][idx]
        
        temp = np.empty(108)
        thisChunk = np.array(ds.isel(x=slice(sI,eI+1)).tmpf.data) # why do I need to convert to np array now? 
        
        if dMapArray['reverse'][idx]:
            thisChunk = np.flip(thisChunk)
            
        temp[0:temp.shape[0]] = np.nan
        temp[0:thisChunk.shape[0]] = thisChunk
        
        NESW[idx-2,:] = temp
        
    for idx in range(18,34):
        sI = dMapArray['idxStart'][idx]
        eI = dMapArray['idxEnd'][idx]
        
        temp = np.empty(108)
        thisChunk = np.array(ds.isel(x=slice(sI,eI+1)).tmpf.data)
        
        if dMapArray['reverse'][idx]:
            thisChunk = np.flip(thisChunk)
            
        temp[0:temp.shape[0]] = np.nan
        temp[0:thisChunk.shape[0]] = thisChunk
        
        SENW[idx-18,:] = temp
        
    return(NESW[:,:107], SENW[:,:107])


def denseAverager(array):
    
    '''
    This just averages the two lowest parts of the dense transect and sends it back
    
    Input: 16 x ~106 dense array
    Output: 1 x 106 array
    '''
    
    array = array[15:16][:]
    array = np.mean(array, axis=0)
    return(array)


def timeSeriesPlotter(array, fname, size=(30,30), detrend=False):
    
    '''
    This plots a DTS time series in a quasi-heatmap.
    Always saves; doesn't show.
    
    Improvements: axis labels, font size, best colour map?
    
    Parameters
    -----------
    array: a 2d array of T(x) vs time. Each column is a DTS transect at a new timestep
    fname: name of the output file - must include full path and extension
    size: optional, moderate plot size
    '''
    
    if len(array.shape) > 2:
        raise ValueError("Incorrect dimensions array, should be 2: %i" % len(array.shape))
        
    fig,ax = plt.subplots(figsize=size)
    
    if detrend:
        high = int(np.nanmax(array))
        low = int(np.nanmin(array))
        limit = max(high, abs(low))
        if limit > 10: limit = 10
        im=ax.imshow(array, interpolation='none', vmax=limit, vmin=-limit, cmap='seismic')
    else:
        im = ax.imshow(array, interpolation='none')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    fig.colorbar(im, cax=cax)
    fig.savefig(fname)
    plt.close(fig)
    
    
def mapArrayPicker():
    
    # TODO: based on time of a chunk or file, pick the appropriate dense.long maps
    
    pass

#%% MAIN --> EXECUTABLE CODE HERE



if __name__ == '__main__':
    
    #%% XML organisation
    # routine for sorting the DTS files in to intelligent 30-minute
    # chunks, reading the .xml files, calibrating the temperatures,
    # and writing out half-hour netCDF-style files
    
    parent = '/Volumes/NamTEX_DATA/NamTEX/DTS/'
    masterFolders = []
    for i in filter(os.path.isdir, glob.glob(os.path.join(parent, '*'))):
        masterFolders.append(i)
    outFolder = '/Volumes/LaCie Drive/DTS_CALIBRATED_TEMPERATURES'
    if not os.path.isdir(outFolder): os.mkdir(outFolder)
    
    for folder in masterFolders[2:]:
        
        print("getting files" + folder)
        #folder = masterFolders[6]
        files = glob.glob(os.path.join(folder, '*.xml'))
        files.sort()
    
        print("getting times")
        times = [getTime(file) for file in files]
    
        timeSequence = getTimeSequence(folder.split('/')[-1])
     
        print("defining chunks")
        chunks = defineChunks(timeSequence, files, fileTimes=times)
    
        f = open(os.path.join(outFolder, folder.split('/')[-1] + '_chunkList.txt'), 'w')
        for idx, chunk in enumerate(chunks):
            f.write('----------------------------\n')
            f.write(' - Chunk number ' + str(idx) + '\n')
        
            if chunk[2] is None:
                f.write("NOT ENOUGH FILES\n")
            else:
                for part in chunk:
                    f.write(str(part) + '\n')
                f.write('Time difference: ' + str(times[chunk[3]]-times[chunk[2]]) + '\n')
                f.write(str(chunk[3] - chunk[2]) + ' files included\n')
        
        f.close()
    
        for chunk in chunks:
        
            if chunk[2] is not None:
                print('------------------------------')
                print(chunk[4])
                dummy = chunkHandler(chunk, files, outFolder)
            else:
                continue
    
    #%% OLD: Full series mapping/plotting
            
    day='1003'
    mapFolder = '/Users/rainerhilland/Documents/PhD/NamTEX/Data/DTS/DTSMaps'
    longMap = glob.glob(os.path.join(mapFolder, '*_Long*'+day+'.csv'))[0] 
    mapArray = pd.read_csv(longMap)
    denseMapArray = pd.read_csv('/Users/rainerhilland/Documents/PhD/NamTEX/Data/DTS/DTSMaps/DTSMap_V02_T02.csv')

    
    # mid-section goes SE -> NW
    # NW long goes Centre -> NW
    # SE long goes SE -> Centre
    # arrange SE_Long, mid, NW_Long; no reversing
    base = '/Volumes/LaCie Drive/DTS_CALIBRATED_TEMPERATURES'
    subFolders = glob.glob(os.path.join(base, '*'))
    subFolders.sort()
    
    for folder in subFolders:
        fname = os.path.join(folder, 'calibratedDTS_DATA.nc')
        if os.path.isfile(fname):
            
            check = os.path.join(folder, 'SEtoNW.png')
            if os.path.isfile(check): 
                print('completed')
                continue
            
            ds = dts.open_datastore(fname)
            
            print(folder)
        
            holder = np.empty((593+106+594+2, len(ds.time))) # these are hardcoded for 1003 map
            holder2 = np.empty((593+106+593+6, len(ds.time)))
            holder_detrend = np.empty((593+106+594+2, len(ds.time)))
            holder2_detrend = np.empty((593+106+593+6, len(ds.time)))
            
            print(len(ds.time))
            for timestep in range(len(ds.time)):
                if timestep % 50 == 0: print(timestep, end='... ')
                thisData = ds.isel(time=timestep)
                
                SEtoNW, SWtoNE = fullMapOneTimestep(mapArray, denseMapArray, thisData, component=False)
                
                SEtoNW_deTrend = np.subtract(SEtoNW, np.nanmean(SEtoNW))
                SWtoNE_deTrend = np.subtract(SWtoNE, np.nanmean(SWtoNE))

                holder[:,timestep] = SEtoNW
                holder2[:,timestep] = SWtoNE
                
                holder_detrend[:,timestep] = SEtoNW_deTrend
                holder2_detrend[:,timestep] = SWtoNE_deTrend

            outname1 = os.path.join(folder, 'SEtoNW.png')
            outname2 = os.path.join(folder, 'SWtoNE.png')
            outname3 = os.path.join(folder, 'SEtoNW_detrend.png')
            outname4 = os.path.join(folder, 'SWtoNE_detrend.png')
            
            timeSeriesPlotter(holder, outname1)
            timeSeriesPlotter(holder2, outname2)
            timeSeriesPlotter(holder_detrend, outname3, detrend=True)
            timeSeriesPlotter(holder2_detrend, outname4, detrend=True)
            
#%% OLD larger maps NOT WORKING
'''
s1 = 593+106+594+2
s2 = 593+106+593+6

for day in range(8,18):
    
    dayString = str(day).zfill(2)
    folders = glob.glob(os.path.join(base, '*.'+dayString+'_*'))
    folders.sort()
    
    master1 = np.empty((s1, 1))
    master2 = np.empty((s2, 1))
    master1[:] = np.nan
    master2[:] = np.nan
    
    for folder in folders[:len(folders)//2]: # half days
        print(folder)
        
        fname = os.path.join(folder, 'calibratedDTS_DATA.nc')
        if os.path.isfile(fname):
            
            ds = dts.open_datastore(fname)
            
            holder = np.empty((s1, len(ds.time)))
            holder2 = np.empty((s2, len(ds.time)))
            
            for timestep in range(len(ds.time)):
                thisData = ds.isel(time=timestep)
            
                SEtoNW, SWtoNE = fullMapOneTimestep(mapArray, denseMapArray, 
                                                    thisData, component=False)
                
                holder[:,timestep] = SEtoNW
                holder2[:,timestep] = SWtoNE
                
        else:
            
            holder = np.empty((s1, 1400))
            holder2 = np.empty((s2, 1400))
            holder[:] = np.nan
            holder2[:] = np.nan
            
        master1 = np.append(master1, holder, axis=1)
        master2 = np.append(master2, holder2, axis=1)
        
        outname1 = os.path.join(base, dayString+'_firstHalfSeriesSEtoNW.png')
        outname2 = os.path.join(base, dayString+'_firstHalfSeriesSWtoNE.png')
        
        factor = master1.shape[1] // master1.shape[0]
        
        timeSeriesPlotter(master1, outname1, size=(16*factor,16))
        timeSeriesPlotter(master2, outname2, size=(16*factor,16))
        
    # second half (just repeats the first)
    
    for folder in folders[len(folders)//2:]: # half days
        print(folder)
        
        fname = os.path.join(folder, 'calibratedDTS_DATA.nc')
        if os.path.isfile(fname):
            
            ds = dts.open_datastore(fname)
            
            holder = np.empty((s1, len(ds.time)))
            holder2 = np.empty((s2, len(ds.time)))
            
            for timestep in range(len(ds.time)):
                thisData = ds.isel(time=timestep)
            
                SEtoNW, SWtoNE = fullMapOneTimestep(mapArray, denseMapArray, 
                                                    thisData, component=False)
                
                holder[:,timestep] = SEtoNW
                holder2[:,timestep] = SWtoNE
                
        else:
            
            holder = np.empty((s1, 1400))
            holder2 = np.empty((s2, 1400))
            holder[:] = np.nan
            holder2[:] = np.nan
            
        master1 = np.append(master1, holder, axis=1)
        master2 = np.append(master2, holder2, axis=1)
        
        outname1 = os.path.join(base, dayString+'_firstHalfSeriesSEtoNW.png')
        outname2 = os.path.join(base, dayString+'_firstHalfSeriesSWtoNE.png')
        
        factor = master1.shape[1] // master1.shape[0]
        
        timeSeriesPlotter(master1, outname1, size=(16*factor,16))
        timeSeriesPlotter(master2, outname2, size=(16*factor,16))
'''
        
#%% OLD FullDay Maps       
# TODO: modify the plotting to INCLUDE TIMES!!!
# DO NOT USE THESE FUNCTIONS
# IMPROVED VERSIONS IN BLM_PROCESSING.py

     
def readMappedArray(fname):
    
    '''TODO: fill, finish the function'''
    
    return(dd.io.load(fname))


def dicToFlatMap(dic, fix=False):
    
    NE = np.swapaxes(dic['NE'], 0, 2)
    NE = np.mean(NE, axis=1)
    
    SW = np.swapaxes(dic['SW'], 0, 2)
    if fix:
        SW = SW[1:,0]
    else:
        SW = np.mean(SW[1:,], axis=1)

    NESW = np.swapaxes(dic['NESW'], 0, 2)
    NESW = NESW[:,14:,:]
    NESW = np.mean(NESW, axis=1)
    NESW = np.flip(NESW, axis=0)
    
    SWtoNE = np.append(SW, NESW, axis=0)
    SWtoNE = np.append(SWtoNE, NE, axis=0)
    
    NW = np.swapaxes(dic['NW'], 0, 2)
    NW = np.mean(NW, axis=1)
    
    SE = np.swapaxes(dic['SE'], 0, 2)
    SE = np.mean(SE, axis=1)
    
    SENW = np.swapaxes(dic['SENW'], 0, 2)
    SENW = SENW[:-1,14:,:]
    SENW = np.mean(SENW, axis=1)
    
    SEtoNW = np.append(SE, SENW, axis=0)
    SEtoNW = np.append(SEtoNW, NW, axis=0)
    
    return(SEtoNW, SWtoNE)


def timeSeriesPlotter2(SENW, SWNE, fname, baseSize=18):
    
    '''
    This plots a DTS time series in a quasi-heatmap.
    Always saves; doesn't show.
    
    Improvements: axis labels, font size, best colour map?
    
    Parameters
    -----------
    array: a 2d array of T(x) vs time. Each column is a DTS transect at a new timestep
    fname: name of the output file - must include full path and extension
    size: optional, moderate plot size
    '''
    
    high = max(np.nanmax(SENW), np.nanmax(SWNE))
    low = min(np.nanmin(SENW), np.nanmin(SWNE))
    
    factor = SENW.shape[1] // SENW.shape[0]
    size = (0.8*baseSize*factor, baseSize*2)
    
    fig,ax = plt.subplots(2,1,figsize=size)
    
    im = ax[0].imshow(SENW, interpolation='none', vmax=high, vmin=low)
    ax[1].imshow(SWNE, interpolation='none', vmax=high, vmin=low)
    
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.25)
    #fig.colorbar(im, ax=ax[1], location='bottom')
    fig.savefig(fname)
    plt.close(fig)
    
    
'''
SEtoNW_Size = 593+106+594+2
SWtoNE_Size = 593+106+593+6

h5Name = 'mappedDTSDictionaries.h5'
for day in range(8,18):
    
    print('#-------------------------------------')
    print(day)
    
    dayString = str(day).zfill(2)
    folders = glob.glob(os.path.join(base, '*.'+dayString+'_*'))
    folders.sort()               
                
    masterSENW = np.empty((SEtoNW_Size, 1))
    masterSWNE = np.empty((SWtoNE_Size, 1))
    masterSENW[:] = np.nan
    masterSWNE[:] = np.nan                
                
    for folder in folders:
        print(folder, end=' ')
        
        fname = os.path.join(folder, h5Name)        
        if os.path.isfile(fname):
            ds = readMappedArray(fname)
            SEtoNW, SWtoNE = dicToFlatMap(ds, fix=True)
            
            while SEtoNW.shape[0] < SEtoNW_Size: # this is pretty ugly
                blankRow = np.repeat(np.nan, SEtoNW.shape[1])
                SEtoNW = np.vstack([SEtoNW, blankRow])
                
            while SWtoNE.shape[0] < SWtoNE_Size:
                blankRow = np.repeat(np.nan, SWtoNE.shape[1])
                SWtoNE = np.vstack([SWtoNE, blankRow])
            
        else:
            print(' ... BLANK', end=' ')
            SEtoNW = np.empty((SEtoNW_Size, 1200))
            SEtoNW[:] = np.nan
            
            SWtoNE = np.empty((SWtoNE_Size, 1200))
            SWtoNE[:] = np.nan
            
        masterSENW = np.append(masterSENW, SEtoNW, axis=1)
        masterSWNE = np.append(masterSWNE, SWtoNE, axis=1)
        print('\n', end=' ')
        
    outName = os.path.join(base, dayString+'_FULL_FIX.png')
    timeSeriesPlotter2(masterSENW, masterSWNE, outName, baseSize=10)                
    
    for folder in folders[:len(folders)//2]:
        print(folder)
        
        fname = os.path.join(folder, h5Name)        
        if os.path.isfile(fname):
            ds = readMappedArray(fname)
            SEtoNW, SWtoNE = dicToFlatMap(ds)
            
            while SEtoNW.shape[0] < SEtoNW_Size: # this is pretty ugly
                blankRow = np.repeat(np.nan, SEtoNW.shape[1])
                SEtoNW = np.vstack([SEtoNW, blankRow])
                
            while SWtoNE.shape[0] < SWtoNE_Size:
                blankRow = np.repeat(np.nan, SWtoNE.shape[1])
                SWtoNE = np.vstack([SWtoNE, blankRow])
            
        else:
            
            SEtoNW = np.empty((SEtoNW_Size, 1200))
            SEtoNW[:] = np.nan
            
            SWtoNE = np.empty((SWtoNE_Size, 1200))
            SWtoNE[:] = np.nan
            
        masterSENW = np.append(masterSENW, SEtoNW, axis=1)
        masterSWNE = np.append(masterSWNE, SWtoNE, axis=1)
        
    outName = os.path.join(base, dayString+'SECONDHALF_fullSeries.png')
    timeSeriesPlotter2(masterSENW, masterSWNE, outName)        
     '''     
                
                
                
                
                
                