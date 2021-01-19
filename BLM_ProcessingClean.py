#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:29:33 2020

@author: rainerhilland

Better organisation of some BLM_Processing functions,
plus documentation :/

Revisions:

    02.12.2020 
        Ubuntu port for extra processing power
        
    08.12.2020
        - merged Ubuntu and mac versions. Should keep only one 'official'
          version of this in use! Folders still hardcoded but there's an OS
          check to determine which to use
        - added some keyword arguments -s and -i to make it easier to jump
          around a 'todo' set of folders
    
    14.12.2020
        - modded to kill SWNE high -> borked

"""

#%% Packages

# --- all imports

# standards
import os, glob, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle, imageio

import warnings#, random
from itertools import compress

# i/o
import deepdish as dd # save/read np arrays
#import pickle, imageio

# some science stuff
from pycurrents.num import spectra
from scipy.stats import linregress
#from sklearn.neighbors import KernelDensity
from scipy import signal#, interpolate
from scipy.optimize import curve_fit

# spectral stuff
from scipy.fftpack import fftfreq, fft
#from pycurrents.num import spectra

# plotting stuff
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib

# dts stuff
import DTSProcessing as dtsPro # my functions
import dtscalibration as dts # calibration routine

# this setting is generally always nice
plt.rcParams['figure.dpi'] = 500

import sys, getopt
# can't remember what this was for
#import pywt
#from pywt.datasetsfrom mpl_toolkits.axes_grid1 import make_axes_locatable as mal

#%% Functions

''' 
These functions are all copied from BLM_Processing and better documented. There
is a lot of work within BLM_Processing that is NOT contained within these
functions - if something's missing go back and find it there.
'''

# --- General DTS Handling/Pre-processing/Chunk work
# --- Old/Probably not necessary

def mapChunk(chunk):
    
    '''
    Given a chunk, this function maps it, i.e. converts the calibrated DTS
    temperatures to individual mapped arrays returned in a dictionary
    
    Part of the calibration/mapping routine
    
    Inputs
    ------
    chunk: a chunk
    
    Outputs
    ------
    arrays: a dictionary with the mapped arrays and the timestamps
    '''
    
    # ehh fuck it take a folder or a file, decide which to settle on later
    if os.path.isfile(chunk):
        if os.path.splitext(os.path.basename(chunk))[1] != '.nc':
            raise TypeError("incompatible file type passed")
        
        ds = dts.open_datastore(chunk)
        
    elif os.path.isdir(chunk):
        fname = os.path.join(chunk, 'calibratedDTS_DATA.nc')
        if os.path.isfile(fname):
            ds = dts.open_datastore(fname)
        else:
            raise ValueError("no calibrated DTS temps here")
    
    else:
        raise TypeError("whatcha doin there bud?")
        
    # get the maps
    startTime = ds.time.data[0]
    longMap, denseMap = mapPicker(startTime)
    timesteps = len(ds.time)
    
    # NB: arrays are in (time, z, x) format!
    arrays = getEmpties(startTime, timesteps)
    
    print("Starting mapping ... ", end='')
    for t in range(timesteps):
        #if t % 5 == 0: print(t, end=' ... ')
        
        thisDat = ds.isel(time=t)
        NW,SE,SW,NE,NESW,SENW = dtsPro.fullMapOneTimestep(longMap, denseMap,
                                                          thisDat, vertMean=False)
        
        arrays['NW'][t] = NW
        arrays['NE'][t] = NE
        arrays['SW'][t] = SW
        arrays['SE'][t] = SE
        arrays['NESW'][t] = NESW
        arrays['SENW'][t] = SENW
        arrays['time'].append(str(thisDat.time.data))
    
    return(arrays)


def getEmpties(time, n):
    
    '''
    Creates dictionary of empty np arrays for use in the mapping process.
    AKA complete clusterfuck of a function in this chain.
    
    Inputs
    ------
    time: start time of the half-hour (for correct map)
    n: number of entries/timestamps
    
    Returns
    ------
    empties: dictionary of empty np arrays of correct dimensions
    '''
    
    split = np.datetime64('2020-03-10 05:30') # need to change if there's a new map
    if time > split:
        empties = {'NW':np.empty((n,2,594)), 'SE':np.empty((n,2,595)), 
                   'SW':np.empty((n,2,596)), 'NE':np.empty((n,2,596)),
                  'NESW':np.empty((n,16,107)), 'SENW':np.empty((n,16,107)),
                  'time':[]}
    elif time < split:
        empties = {'NW':np.empty((n,2,593)), 'SE':np.empty((n,2,594)), 
                   'SW':np.empty((n,2,593)), 'NE':np.empty((n,2,593)),
                  'NESW':np.empty((n,16,107)), 'SENW':np.empty((n,16,107)),
                  'time':[]}
    else:
        raise ValueError("you somehow broke something that you shouldn't have")
        
    return(empties)


def mapPicker(time, folderOverride=None, 
              denseOverride=None, longOverride=None):
    
    '''
    Picks the right map to use for a given time/chunk. These are the
    *MAPPING MAPS* that define how to take the raw DTS output and put it in
    to NamTEX coordinates.
    
    As of 2020.10.20 only using 2 maps (which I hope will suffice). There
    are 2 long maps which split at 2020.03.10 05:30 and one dense map.
    
    You only need to provide a timestamp to use this default approach,
    otherwise you can use the other parameters to override the default
    behaviour.
    
    Parameters
    -----------
    time: np.datetime64 format timestamp
    folderOverride: if you want to look somewhere other than the default
                    DTSMaps folder
    denseOverride: you can pass a map specifically here to override the default
    longOverride: you can pass a map specifically here to override the default
    
    Returns
    -------
    longMapArray: the map for the long sections (pd dataframe)
    denseMapArray: the map for the dense sections (pd dataframe)
    
    '''
    
    if not isinstance(time, np.datetime64):
        raise TypeError("input time needs to be np.datetime64")
    
    # big break is repaired and DTS comes online 2020-03-10 05:28
    split = np.datetime64('2020-03-10 05:30')
    
    if time > split: # after break use 10-03 long map
        mapDay = '1003'
    elif time < split:
        mapDay = '0803'
    else:
        raise ValueError("not able to pick a day - check the time")
        
    if folderOverride is None:
        global OS
        if OS == 'linux':
            folder = '/home/rainer/dev/DTSMaps'
        elif OS == 'darwin':
            folder = '/Users/rainerhilland/Documents/PhD/NamTEX/Data/DTS/DTSMaps'
    else:
        folder = folderOverride
        
    if longOverride is None:
        longMap = glob.glob(os.path.join(folder, '*_Long*'+mapDay+'.csv'))[0]
        longMapArray = pd.read_csv(longMap)
    else:
        longMapArray = pd.read_csv(longOverride)
        
    if denseOverride is None:
        denseMap = glob.glob(os.path.join(folder, '*V02*.csv'))[0]
        denseMapArray = pd.read_csv(denseMap)
        
    return(longMapArray, denseMapArray)


def interpMapPicker(time):
    
    '''
    Picks the right map for a given timestamp/chunk. *INTERPOLATION MAPS* which
    define how to interpolate out the hot spots in the array.
    
    NB: at the moment (2020.10.20) this only covers the long sections. NO
    interpolation exists for the dense section yet.
    
    Parameters
    -----------
    time: timestamp to use
    
    Returns
    --------
    the interpolation dictionary (as pd dataframe)
    '''
    
    split = np.datetime64('2020-03-10 05:30')
    time = np.datetime64(time)
    
    if time > split:
        return(getInterpDictionary(day='1003'))
    if time < split:
        return(getInterpDictionary(day='0803'))
    else:
        return ValueError("I'm tired of writing intelligent errors")


def saveMappedArray(arrayDictionary, fname):
    
    '''
    part of the dts pre-processing mapping routine. Saves a mapped dictionary
    in a useful way using deepdish
    
    Parameters
    ----------
    arrayDictionary: a dictionary of mapped arrays
    fname: the name to save as. As far as I know extension doesn't matter
    '''
    
    if not isinstance(arrayDictionary, dict):
        raise TypeError("y'done fucked up, son")
        
    dd.io.save(fname, arrayDictionary)


def dicToFlatMap(dic, fix=False, TCorr=False):
    
    '''
    UPDATE: 2020.11.23 --> DO NOT USE
     -> this was useful for plotting but is not useful for analysis. For 
         similar tasks use the functions below:
             getmeafullday_oneheight()
             getmeoneheight()
    
    FYI this is returning a map mostly useful for plotting,
    hence the swapaxes stff
    
    temperature correction as of 23.10.2020 assumes that the middle
    part as the most accurate mean (closer to cable start; no poles)
    
    I might deprecate this...
    
    Parameters
    -----------
    dic: the array dictionary with the data
    fix: T/F, will remove the lower height from the SW axis if True
    TCorr: T/F will balance the means of the long sections against the dense
        if set to True
        
    Returns
    --------
    dicOut: a dictionary with SENW and SWNE arrays
    '''
    
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
    
    if TCorr:
        referenceMean= np.mean(NESW)
        NEMean = np.mean(NE)
        SWMean = np.mean(SW)
    
        NE = np.subtract(NE, NEMean-referenceMean)
        SW = np.subtract(SW, SWMean-referenceMean)
    
    SWtoNE = np.append(SW, NESW, axis=0)
    SWtoNE = np.append(SWtoNE, NE, axis=0)
    
    NW = np.swapaxes(dic['NW'], 0, 2)
    NW = np.mean(NW, axis=1)
    
    SE = np.swapaxes(dic['SE'], 0, 2)
    SE = np.mean(SE, axis=1)
    
    SENW = np.swapaxes(dic['SENW'], 0, 2)
    SENW = SENW[:-1,14:,:]
    SENW = np.mean(SENW, axis=1)
    
    if TCorr:
        referenceMean = np.mean(SENW)
        NWMean = np.mean(NW)
        SEMean = np.mean(SE)
    
        NW = np.subtract(NW, NWMean-referenceMean)
        SE = np.subtract(SE, SEMean-referenceMean)
    
    SEtoNW = np.append(SE, SENW, axis=0)
    SEtoNW = np.append(SEtoNW, NW, axis=0)
    
    dicOut = {'SENW':SEtoNW, 'SWNE':SWtoNE}
    
    return(dicOut)


def getMeAFullDay_OneHeight(day, height):
    
    '''is a full day going to be useful?'''
    
    base = '/Volumes/LaCie Drive/DTS_CALIBRATED_TEMPERATURES' # default for now
    
    folders = glob.glob(os.path.join(base, '*.'+str(day).zfill(2)+'_*'))
    folders.sort()
    
    if height <= 2:
        
        pass # I'm not sure this is useful
    

def interpolateChunk(data, interpMaps=None, p=6):
    
    '''
    This function handles the interpolation of a single 30-min chunk.
    
    NB: interpolated dictionaries are saved as one of the .h5 files for 
        each 30-min section. If you re-calibrate this could be useful again
        but there's no need to re-interpolate the 30-minute chunks

    NB2: 2020.10.20: ONLY LONG SECTIONS are interpolated
    
    ** WARNING **: by default this uses the 10.03 interpolater.
    
    Inputs
    --------
    data: a mapped DTS dictionary
    interpMaps: optional, a dictionary of the interpolation maps. Defaults to 10.03
    p: how many points to use for fitting the interpolation, default 6
    
    Returns
    --------
    newDat: dictionary with same keys and dimensions as the input, but interpolated
    '''
        
    if not isinstance(data, dict):
        return TypeError('data should be a mapped DTS dictionary bro')
    
    if interpMaps is None:
        interpMaps = getInterpDictionary()
        #TODO: incorporate day selection
        
    newDat = data.copy() # let's be more responsible
    timesteps = len(newDat['time'])
    
    print("interpolating (%i timesteps) ... " % timesteps, end='')
    for t in range(timesteps):
        
        if t % 250 == 0: print('%i ...' % t, end='')
        
        newDat['NE'][t,0,:] = interpolateSingleTransect(newDat['NE'][t,0,:], interpMaps['NEHigh'], p=p)
        newDat['NE'][t,1,:] = interpolateSingleTransect(newDat['NE'][t,1,:], interpMaps['NELow'], p=p)
        
        newDat['NW'][t,0,:] = interpolateSingleTransect(newDat['NW'][t,0,:], interpMaps['NWHigh'], p=p)
        newDat['NW'][t,1,:] = interpolateSingleTransect(newDat['NW'][t,1,:], interpMaps['NWLow'], p=p)
        
        newDat['SE'][t,0,:] = interpolateSingleTransect(newDat['SE'][t,0,:], interpMaps['SEHigh'], p=p)
        newDat['SE'][t,1,:] = interpolateSingleTransect(newDat['SE'][t,1,:], interpMaps['SELow'], p=p)
        
        newDat['SW'][t,0,:] = interpolateSingleTransect(newDat['SW'][t,0,:], interpMaps['SWHigh'], p=p)
        newDat['SW'][t,1,:] = interpolateSingleTransect(newDat['SW'][t,1,:], interpMaps['SWLow'], p=p)
        
    print('')
    return(newDat)

        
def interpolateSingleTransect(ts, interpMap, p=6):
    
    '''
    Sister function of interpolateChunk which handles the interpolation.
    '''
    newts = np.copy(ts) # let's not modify the input array?
    for idx, row in interpMap.iterrows():
        #print(idx)
        startIdx = row['startIdx']
        endIdx = row['endIdx']
        note = row['note']
        newts[startIdx:endIdx] = interpolateSlice(ts, startIdx, endIdx, note, p=p)
        
    return(newts)
    

def interpolateSlice(ts, idxStart, idxEnd, note='nan', p=3):
    
    '''
    Sister-sister function of interpolateChunk. This does the actual work.
    See interpolateChunk and interpolateSingleTransect for deets.
    '''
    
    if note == 'start': # interpolate starting points using following p points
        '''    
        iLength = idxEnd-idxStart
        x = np.arange(idxEnd+1, idxEnd+p+1)
        y = ts[idxEnd:idxEnd+p] 
        
        tck = interpolate.splrep(x,y,s=0,k=1)
        xnew = np.arange(idxStart+1,idxEnd+1)
        ynew = interpolate.splev(xnew, tck, der=0)
        if len(ynew) == iLength:
            return(ynew)
        else:
            print("Something dun fucked up hurr (start error)")   
        '''
        n = np.arange(idxEnd-idxStart) # other approach giving really bad results
        n[:] = ts[idxEnd+1]
        n = n + np.random.normal(0,1,len(n)) # add noise (see write-up)
        return(n)
 
    elif note == 'end':
        '''
        iLength = idxEnd-idxStart
        x = np.arange(1,p+1)
        y = ts[idxStart-p:idxStart]
        
        tck = interpolate.splrep(x,y,s=0,k=1)
        xnew = np.arange(p+1,p+1+iLength)
        ynew = interpolate.splev(xnew, tck, der=0)
        if len(ynew) == iLength:
            return(ynew)
        else:
            print("Ye've made a grave mistake lad (end error)")
        '''
        n = np.arange(idxEnd-idxStart)
        n[:] = ts[idxStart-1]
        n = n + np.random.normal(0,1,len(n))
        return(n)
    
    elif np.isnan(note): # interpolate between S:E +/- p
        iLength = idxEnd-idxStart # interpolation length
        x = np.arange(1, p+1)
        x = np.append(x, np.arange(p+1+iLength, p+1+iLength+p))
        y = ts[idxStart-p:idxStart]
        y = np.append(y, ts[idxEnd:idxEnd+p])
        '''
        tck = interpolate.splrep(x,y) # no smoothing
        xnew = np.arange(p+1,p+1+iLength)
        ynew = interpolate.splev(xnew, tck, der=0)
        '''
        z = np.polyfit(x,y,3)
        interp = np.poly1d(z)
        
        xnew = np.arange(p+1,p+1+iLength)
        ynew = [interp(m) for m in xnew]
        ynew = ynew + np.random.normal(0,1,iLength) # try to add some noise?
        if len(ynew) == iLength:
            return(ynew)
        else:
            print("Something dun fucked up here (nan error)")
            
    else:
        raise ValueError("Invalid 'note' given")

    
def buildInterpDictionary(folder=None, day='1003'):
    
    '''
    Intermediary function that compiles individual CSV interpolation maps
    in to a dictionary of pd frames.
    '''
    
    global OS
    
    if folder is None:
        if OS == 'darwin':
            folder = '/Users/rainerhilland/Documents/PhD/NamTEX/Data/DTS/DTSMaps'
        elif OS == 'linux':
            folder = '/home/rainer/dev/DTSMaps'
        
    inFiles = glob.glob(os.path.join(folder, 'Interp*'+day+'*'))
    interpDic = {}
    
    for file in inFiles:
        loc = os.path.basename(file).split('_')[1]
        interpMap = pd.read_csv(file, sep=';')
        interpDic[loc] = interpMap
        
    return(interpDic)


def getInterpDictionary(folder=None, day='1003'):
    
    '''
    Reads an interpolation dictionary for use in interpolation.
    '''
    
    if folder is None:
        global OS
        if OS == 'darwin':
            folder = '/Users/rainerhilland/Documents/PhD/NamTEX/Data/DTS/DTSMaps'
        elif OS == 'linux':
            folder = '/home/rainer/dev/DTSMaps'
        
    fname = glob.glob(os.path.join(folder, '*'+day+'.pkl'))[0]
    print(' using %s' % fname)
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
    
def balanceMeans(data):
    
    '''
    Balances the means across an entire transect. Reference mean is always
    the centre. Input data should already be interpolated and/or masked
    
    ** non-destructive **
    
    Input
    ------
    Interpolated/masked dictionary
    
    Output
    ------
    New dictionary with balanced means
    '''
    
    print('balancing means ... ')
    ds = data.copy()
    
    # part 1: SENW
    # LOW:
    referenceMean = np.nanmean(ds['SENW'][:,-1,:])
    SEMean = np.nanmean(ds['SE'][:,1,:])
    NWMean = np.nanmean(ds['NW'][:,1,:])
    
    ds['SE'][:,1,:] = np.subtract(ds['SE'][:,1,:], SEMean-referenceMean)
    ds['NW'][:,1,:] = np.subtract(ds['NW'][:,1,:], NWMean-referenceMean)
    
    # HIGH:
    referenceMean = np.nanmean(ds['SENW'][:,-2:-1,:])
    SEMean = np.nanmean(ds['SE'][:,0,:])
    NWMean = np.nanmean(ds['NW'][:,0,:])
    
    ds['SE'][:,0,:] = np.subtract(ds['SE'][:,0,:], SEMean-referenceMean)
    ds['NW'][:,0,:] = np.subtract(ds['NW'][:,0,:], NWMean-referenceMean)

    # part 2: NESW
    # LOW:
    referenceMean = np.nanmean(ds['NESW'][:,-1,:])
    SWMean = np.nanmean(ds['SW'][:,1,:150])
    NEMean = np.nanmean(ds['NE'][:,1,:])
    
    ds['SW'][:,1,:] = np.subtract(ds['SW'][:,1,:], SWMean-referenceMean)
    ds['NE'][:,1,:] = np.subtract(ds['NE'][:,1,:], NEMean-referenceMean)
    
    # HIGH:
    referenceMean = np.nanmean(ds['NESW'][:,-2:-1,:])
    SWMean = np.nanmean(ds['SW'][:,0,:])
    NEMean = np.nanmean(ds['NE'][:,0,:])
    
    ds['SW'][:,0,:] = np.subtract(ds['SW'][:,0,:], SWMean-referenceMean)
    ds['NE'][:,0,:] = np.subtract(ds['NE'][:,0,:], NEMean-referenceMean)
    
    return(ds)


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

        
# --- Current/Useful Handling/Pre-proc

def cleanUp(folder, of):
    files = glob.glob(os.path.join(folder, of))
    dummy = [os.remove(f) for f in files]
    print('cleaned %i %s' % (len(files), of))


def readMappedArray(fname):
    
    '''
    Reads a mapped array!
    
    Parameters
    ----------
    fname: the filename to open
    
    Returns
    --------
    a dictionary of mapped arrays
    
    Remember: dictionary is arranged with the following keys
    NE
    NESW
    NW
    SE
    SENW
    SW
    time -> just the strings
    time_formatted -> np.datetime64 objects
    
    and with shapes (time, height, distance) or (t,z,x)
    REMEMBER that heights are backwards (kind of)
    0 = highest height, higher numbers are lower
    '''
    
    f = '%Y-%m-%dT%H:%M:%S.%f'
    inDic = dd.io.load(fname)
    inDic['time_formatted'] = [datetime.strptime(x[:-3], f) for x in inDic['time']]
    
    return(inDic)


def getmeoneheight(dic, height, side, mask, dense=False):
    
    '''
    'SENW', 'SWNE', 'both'
    given a mapped dictionary where keys are component pieces, return
    a combined array for a given height
    
    only used for heights 0/1 at the moment (dense section is easy to just
                                             subset however you want)
    TODO: add an interp for the dense section
    
    mask = T/F --> mask the arrays?
    
    Parameters
    ----------
    dic: a dictionary of mapped arrays
    height: the height you want returned (int)
    side: string. 'SENW', 'SWNE' or 'both'
    
    Returns
    --------
    1-2 larger arrays containing the full extent at height=height
    '''

    if height == 0:
        denseHeight = 14
    elif height == 1:
        denseHeight = 15
    elif not dense:
        raise ValueError('set the dense flag so there are no height cock-ups')
        
    if mask: 
        intDic = getInterpDictionary() # add override for this
        hKeyL = 'High' if height == 0 else 'Low'
        
    if dense:
        if side == 'SENW':
            
            SENW = dic['SENW'][:,DHT(height),:-1]
            if mask: SENW = maskSingleArray(SENW, intDic['SENW'])
            return(SENW)
        
        elif side == 'SWNE':
            
            SWNE = np.flip(dic['NESW'][:,DHT(height),:], axis=1)
            if mask: SWNE = maskSingleArray(SWNE, intDic['SWNE'])
            return(SWNE)
            
        elif side == 'both':
            raise ValueError("no 'both' for dense, I'm too lazy")
        else: raise ValueError('invalid side: SENW or SWNE')
    
    if side == 'SENW':
        
        NW = dic['NW'][:,height,:]
        SE = dic['SE'][:,height,:]
        SENW = dic['SENW'][:,denseHeight,:-1]
        
        if mask:
            NW = maskSingleArray(NW, intDic['NW'+hKeyL])
            SE = maskSingleArray(SE, intDic['SE'+hKeyL])
            SENW = maskSingleArray(SENW, intDic['SENW'])
        
        full = np.append(SE,SENW, axis=1)
        full = np.append(full, NW, axis=1)
        
        return(full)
    
    elif side == 'SWNE':
        
        NE = dic['NE'][:,height,:]
        SW = dic['SW'][:,height,:]
        SWNE = np.flip(dic['NESW'][:,denseHeight,:], axis=1)
        
        if mask:
            NE = maskSingleArray(NE, intDic['NE'+hKeyL])
            SW = maskSingleArray(SW, intDic['SW'+hKeyL])
            SWNE = maskSingleArray(SWNE, intDic['SWNE'])
        
        full = np.append(SW, SWNE, axis=1)
        full = np.append(full, NE, axis=1)
        
        # SKIP THE FUCKED UP SECTION!
        full = full[:,500:]
        
        return(full)
    
    elif side == 'both':
        
        NW = dic['NW'][:,height,:]
        SE = dic['SE'][:,height,:]
        SENW = dic['SENW'][:,denseHeight,:-1]
        
        fullSENW = np.append(SE,SENW, axis=1)
        fullSENW = np.append(fullSENW, NW, axis=1)
    
        NE = dic['NE'][:,height,:]
        SW = dic['SW'][:,height,:]
        SWNE = np.flip(dic['NESW'][:,denseHeight,:], axis=1)
        
        fullSWNE = np.append(SW, SWNE, axis=1)
        fullSWNE = np.append(fullSWNE, NE, axis=1)
        
        return(fullSENW, fullSWNE)
    
    else:
        raise ValueError('invalid "side" argument: %s' % side)


def STavg(data, space, time, sFactor, tFactor):
    
    '''
    Performs spatio-temporal averaging of a DTS (or I guess any other) array.
    
    -> working for now
    
    doing some spatio-temporal averaging
        
    space/time: T/F do space/time averaging?
    sFactor/tFactor: integer, by which factor to reduce?
        i.e. factor=2 averages two points together, reduces size by 2
             factor=3 averages three points together, reduces size by 3
             
    idxs don't always evenly divide and so this function DROPS DATA AT THE END!
    
    Inputs
    -------
    data: (t,x) input array
    space/time: T/F whether or not to average in space/time
    s/t Factor: factor by which to average
    '''
    
    if len(data.shape) > 2:
        raise ValueError("too many dims sent to STavg")
        
    timeShape, spaceShape = data.shape
    
    if space and time:
        newDat1 = np.empty((timeShape, spaceShape//sFactor))
        
        print("avging in space")
        # avg in space
        for counter, idx in enumerate(range(0, spaceShape, sFactor)):
            if counter == spaceShape//sFactor: continue
            crop = data[:,idx:idx+sFactor]
            newDat1[:,counter] = np.nanmean(crop, axis=1)
        
        newDat = np.empty((timeShape//tFactor, spaceShape//sFactor))
        print("avging in time")
        # avg in time
        for counter ,idx in enumerate(range(0, timeShape, tFactor)):
            if counter == timeShape//tFactor: continue
            crop = newDat1[idx:idx+tFactor,:]
            newDat[counter,:] = np.nanmean(crop, axis=0)
            
    if space and not time:
        newDat = np.empty((timeShape, spaceShape//sFactor))
        
        print("avging in space only")
        # avg in space
        for counter, idx in enumerate(range(0, spaceShape, sFactor)):
            if counter == spaceShape//sFactor: continue
            crop = data[:, idx:idx+sFactor]
            newDat[:,counter] = np.nanmean(crop, axis=1)
            
    if time and not space:
        newDat = np.empty((timeShape//tFactor, spaceShape))
        
        print("avging in time only")
        # avg in time
        for counter, idx in enumerate(range(0, timeShape, tFactor)):
            if counter == timeShape//tFactor: continue
            crop = data[idx:idx+tFactor,:]
            newDat[counter,:] = np.nanmean(crop, axis=0)

    return(newDat)

      
def timeChopDTS(data, nSegments, times=None):
    
    '''
    Chops a full DTS array in to n equal segments in time. Just used for 
    initial playing around so far, should make more dynamic at some point.
    
    NB: probably drops some time steps in the end because of the //
    
    Inputs
    -------
    data: (t,x) DTS array
    nSegements: how many segments to chop the array in to
    times: not used yet. Would improve things.
    '''
    
    if isinstance(data, dict):
        raise ValueError("for now you can only pass np arrays, not dicts")
        
    if (x:=len(data.shape)) > 2:
        raise ValueError("array should be (t,x), i.e. only one height: %i" % x)
    
    nTimes = data.shape[0]
    segSize = nTimes // nSegments
    
    segIndices = []
    for idx in range(nSegments+1):
        segIndices.append(idx*segSize)
        
    segments = {}
    for idx in range(len(segIndices)-1):
        sI = segIndices[idx]
        eI = segIndices[idx+1]
        subSeg = data[sI:eI,:]
        segments[str(idx).zfill(2)] = subSeg
        
    print("Reduced %i times in to %i segments of size %i" % (nTimes, nSegments, segSize))
    
    return(segments)


def iterEachPoint(data, func):
    
    '''
    Simple function that iterates over every point in the array and applies
    some function to it. It can/may be useful in assessing homogeneity of
    temperature across the array.
    
    Inputs
    ------
    data: either a cropped array OR a mapped dictionary.
    func: the function to be applied, e.g. np.nanmean, np.sd, etc.
    
    Returns
    -------
    holder: a list of the function outputs
    heights: if a mapped dict is passed, the heights at which values occur
    '''
    
    if isinstance(data, dict):
        
        holder = []
        heights = []
        for key in data:
            
            if key == 'time' or key == 'time_formatted':
                continue
            
            subData = data[key]
            _, Hs, Xs = subData.shape
            
            for x in range(Xs):
                for h in range(Hs):
                    heights.append(h)
                    holder.append(func(subData[:,h,x]))
                    
        return(heights, holder)
        
    elif isinstance(data, np.ndarray) and len(data.shape) == 2:
        
        _, Xs = data.shape
        
        holder = []
        for x in range(Xs):
            holder.append(func(data[:,x]))
        
        return(holder)
    
    else:
        raise ValueError('data should be dict or 2d np array')
        
        
def dtsDetrend(d):
    '''detrends each point. Used for calibration bath stuff
    ** destructive **'''
    for p in range(d.shape[1]):
        d[:,p] = signal.detrend(d[:,p])
    return(d)


def getClasses(override=None):
    
    '''
    gets a classification file. 
    
    2020.11.24 -> new 5-min flux file set as default, using the 3m sonic (no
                  IRGA fluxes) and the smallest possible z0 in eddy pro of
                  0.01 m. Doesn't include the stationarity stuff from before
                  but that can be re-done later.
               -> the time_formatted column is modified to 
    
    2020.11.23 - Default is the 5-min classification 
    'fulltable' on the NamTEX1 drive. This can be overriden w/ the override 
    parameter.
    
    Returns a pd dataframe for add'tl subsetting
    '''
    
    # old table
    # '/Volumes/NamTEX_DATA/NamTEX/SonicTower/5MinClassifications/fullTable.csv'
    
    if override is None:        
        f = '/Users/rainerhilland/Documents/PhD/NamTEX/Data/SonicTower/'
        f += '2020.11.24_Fluxes_Newz0/EPro_Output/3Metre_5Min_Newz0_FLUXES_clean.csv'
        dat = pd.read_csv(f)
    else:
        dat = pd.read_csv(override)
        
    # to keep other functions functioninining
    if 'wind_dir' in dat:
        dat = dat.rename(columns={'wind_dir':'avgWindDir'})
        
    if 'dt_strings' in dat:
        dat = dat.rename(columns={'dt_strings':'timestamp'})
        f = '%Y-%m-%d %H:%M'
    else:
        f = '%Y-%m-%d %H:%M:%S'
        
    
    dat['timestamp_formatted'] = [datetime.strptime(x, f) - timedelta(minutes=5)
                                  for x in dat['timestamp']]
    
    return(dat)


def windDirCrop(table, low, high, fullReturn=False):
    
    '''crop the classification table to between a, b. If fullReturn then
    the entire table is returned, if false only the timestamps
    
    ** NB: As per EddyPro documentation, the timestamps are the ENDS of the
            intervals **
            
    '''
    
    if fullReturn:
        return(table[(table['avgWindDir'] > low) & (table['avgWindDir'] < high)])
    else:
        return(table['timestamp'][(table['avgWindDir'] > low) &
                                  (table['avgWindDir'] < high)])
    

def getDTSData(startTime, endTime=None, precise=True, processed=True):
    
    '''
    right now times can't cross 30-min boundary. Will still work regardless,
    but will only go to the end of the half hour.
    
    Be smart: make sure end time > start time
    
    --> extend past the 30-min only if it's really worth the effort. It will
        be a lot of work
    
    Parameters
    ----------
    startTime: datetime or Timestamp object
    endTime: ditto, optional
    precise: T/F - crop to start/end times or not?
    processed: if True, reads the interpolated and mean-balanced h5
    
    '''
    
    folder = getDTSFolder(startTime)
    
    if not os.path.isdir(folder):
        raise ValueError("no DTS data here: %s" % folder)
        
    if os.path.isfile(os.path.join(folder, 'note.txt')):
        raise ValueError("broken cable time :(")
    
    files = glob.glob(os.path.join(folder, '*.h5'))
    files.sort()
    
    if processed:
        inDat = readMappedArray(files[1])
    else:
        inDat = readMappedArray(files[0])
        
    if precise:
        validStart = nearestTime(startTime, inDat['time_formatted'])
        startIndex = inDat['time_formatted'].index(validStart)
    else:
        startIndex = 0
    if precise or endTime is not None:
        validEnd = nearestTime(endTime, inDat['time_formatted'])
        endIndex = inDat['time_formatted'].index(validEnd) + 1
    else:
        endIndex = len(inDat['time_formatted'])
        
    for key in inDat:
        if isinstance(inDat[key], np.ndarray):
            inDat[key] = inDat[key][startIndex:endIndex,:,:]
        elif isinstance(inDat[key], list):
            inDat[key] = inDat[key][startIndex:endIndex]
            
    return(inDat)
    
def TSList():
    
    ''' I don't think this actually matters '''
    
    f = '/Users/rainerhilland/DTS_Fork'
    folders = glob.glob(os.path.join(f, '03.*'))
    folders.sort()
    
    f = '%m.%d_%H%M'
    t = [x.split('/')[-1] for x in folders]
    tStamps = [datetime.strptime(x, f) for x in t]
    tStamps = [x.replace(year=2020) for x in tStamps]
    
    return(tStamps)


def getDTSFolder(time):
    
    if not isinstance(time, datetime):
        try:
            time = time.to_pydatetime()
        except:
            raise ValueError("time needs to be datetime or Timestamp")
    
    if time.minute < 30:
        time = time.replace(minute=0)
    else:
        time = time.replace(minute=30)
    
    folderName = ('03.' + str(time.day).zfill(2) + '_' + 
                  str(time.hour).zfill(2) + str(time.minute).zfill(2))
    
    base = '/Users/rainerhilland/DTS_Fork'
    return(os.path.join(base, folderName))


def nearestTime(ts, targets):
    
    return(min(targets, key=lambda t: abs(ts-t)))


def saveFrame(frame, name):
    
    '''saving pd data frames'''
    
    if os.path.splitext(name)[1] != '.pkl':
        name = ''.join([os.path.splitext(name)[0], '.pkl'])
        
    frame.to_pickle(name)
    
    
def readFrame(name):
    
    return(pd.read_pickle(name))

# --- convective vel. OLD

def fullCorrOnePoint(inputData, testPointIndex, limitDistance=True, 
                     distance=50, symmetry=False):
    
    '''
    This runs the 2-point correlation routine on a single point, i.e. tests
    one point (testPointIndex) against all others.
    
    ** OLD, DO NOT USE **
        -> deprecated as this Frankenstein is unnecessarily complex
    '''
    
    # for now: only pass single heights
    if len(inputData.shape) > 2:
        raise ValueError("input array is not the correct dimensions")
        
    # always referenced against this point
    refTimeSeries = inputData[:,testPointIndex]
    
    # data holders
    # e.g. distance_m and distance_m_abs don't both need to be stored?
    corrCoeffList = []
    timeLag_idxList = []
    timeLag_idxList_Gauss = [] #
    distance_idxList = []
    distance_metres_absList = []
    distance_metresList = []
    timeLag_sList = []
    timeLag_sList_Gauss = [] #
    abs_velocityList = []
    abs_velocityList_Gauss = [] #
    velocityList = []
    velocityList_Gauss = [] #
    
    if limitDistance:
        totalIdxs = int(distance // .254)
        start = testPointIndex - (totalIdxs//2)
        if start < 0: start = 0
        end = testPointIndex + (totalIdxs//2)
        if end > inputData.shape[1]: end = inputData.shape[1]
    else:
        start = 0
        end = inputData.shape[1]
        
    for idx in range(start,end):
        #print(idx)
        #if idx % 25 == 0: print(idx, end= ' ... ')
        
        testTimeSeries = inputData[:,idx]
        if np.isnan(testTimeSeries[0]):
            corrCoeff = np.nan
            timeLag_idx = np.nan
            timeLag_idx_Gauss = np.nan #
            distance_idx = idx - testPointIndex
            distance_metres_abs = abs(distance_idx * 0.254)
            distance_metres = distance_idx * 0.254
            timeLag_s = np.nan
            timeLag_s_Gauss = np.nan #
            derived_abs_velocity = np.nan
            derived_abs_velocity_Gauss = np.nan #
            derived_velocity = np.nan
            derived_velocity_Gauss = np.nan #
            
        else:   
            corrCoeff, timeLag_idx, timeLag_idx_Gauss = singleLag(refTimeSeries, testTimeSeries)
            distance_idx = idx - testPointIndex
            if not symmetry and distance_idx < 0: continue
            distance_metres_abs = abs(distance_idx * 0.254) # and pray your mapping is OK
            distance_metres = distance_idx * 0.254
            timeLag_s = abs(timeLag_idx * 1.288)
            timeLag_s_Gauss = abs(timeLag_idx_Gauss * 1.288)
            if timeLag_s == 0:
                derived_abs_velocity = np.nan
                derived_velocity = np.nan
                derived_abs_velocity_Gauss = np.nan
                derived_velocity_Gauss = np.nan
            else:
                derived_abs_velocity = distance_metres_abs / timeLag_s
                derived_velocity = distance_metres / timeLag_s
                derived_abs_velocity_Gauss = distance_metres_abs / timeLag_s_Gauss
                derived_velocity_Gauss = distance_metres / timeLag_s_Gauss
                    
        corrCoeffList.append(corrCoeff)
        timeLag_idxList.append(timeLag_idx)
        timeLag_idxList_Gauss.append(timeLag_idx_Gauss) #
        distance_idxList.append(distance_idx)
        distance_metresList.append(distance_metres)
        distance_metres_absList.append(distance_metres_abs)
        timeLag_sList.append(timeLag_s)
        timeLag_sList_Gauss.append(timeLag_s_Gauss)
        abs_velocityList.append(derived_abs_velocity)
        abs_velocityList_Gauss.append(derived_abs_velocity_Gauss)
        velocityList.append(derived_velocity)
        velocityList_Gauss.append(derived_velocity_Gauss)
        
    return({'corrCoeff':corrCoeffList, 'timeLag_idx':timeLag_idxList,
            'distance_idx':distance_idxList, 'distance_m_abs':distance_metres_absList,
            'timeLag_s':timeLag_sList, 'abs_velocity':abs_velocityList,
            'distance_m':distance_metresList, 'velocity':velocityList,
            'timeLag_idx_Gauss':timeLag_idxList_Gauss,
            'timeLag_s_Gauss':timeLag_sList_Gauss,
            'abs_velocity_Gauss':abs_velocityList_Gauss,
            'velocity_Gauss':velocityList_Gauss})


# --- convective velocity functions


def shiftSignal(arr, num, fill_value=np.nan):
    
    '''Shifts array arr by num indices and fills with fill_value'''
    
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def maskSingleArray(array, dic):
    
    '''
    Mask an array. Replaces non-desired points w/ np.nan
    
    Inputs
    -------
    array: a single (t,x) DTS array
    dic: a pd dataframe interpolation map. points that would be interpolated
            are those which get masked
    
    Returns
    -------
    newArr: a masked array
    '''
    
    newArr = np.copy(array)
    
    for idx in range(0,newArr.shape[1]):
        if not checkPoint(idx, dic):
            newArr[:,idx] = np.nan
        else:
            continue
    return(newArr)


def checkPoint(idx, dic):
    
    '''
    Checks indices against a dic to see if they're usable or not. Part
    of array masking (maskSingleArray())
    '''
    
    for i, row in dic.iterrows():
        
        if idx >= row['startIdx']:
            if idx < row['endIdx']:
                return(False)
        
    return(True)


def singleLag(ts1, ts2):
    
    '''
    Calculates the cross-correlation between two time series and returns
    the time lag associated w/ the maximum correlation coefficient.
    
    Part of the convective velocity two-point corr. routine
    
    NB: time lags are in INDICES. i*1.288 for seeconds.
    
    Part of fullCorrOneHeight()
    
    # fyi lags are 
    
    Inputs
    ------
    ts1/ts2: two time series to correlate
    
    Outputs
    ------
    maxCorr: the max normalised correlation coefficient, [-1,1]
    timeLag: the index associated with the max corr
    timeLagGauss: the index associated with the max corr of a gaussian fit
            through the peak of correlation
    '''
    
    # detrend the signals so that 0-padding behaves
    ts1_m = signal.detrend(ts1)
    ts2_m = signal.detrend(ts2)
    
    #denom = np.sqrt(np.std(ts1_m) * np.std(ts2_m))
       
    nx = len(ts1)
    lags = np.arange(-nx+1, nx)
    
    # this variable is named wrong. Result is not autocorrelation.
    autocorr = np.correlate(ts1_m, ts2_m, mode='full')
    autocorr *= 1/len(ts1)
    autocorr /= np.sqrt(np.var(ts1_m) * np.var(ts2_m))
  
    maxCorr = np.max(autocorr) # maximum correlation value
    i = np.argmax(autocorr)
    timeLag = lags[i] # corresponding time-lag
    
    # the Gaussian fitting stuff
    subLag = lags[i-1:i+2]
    subCorr = autocorr[i-1:i+2]
    newX = np.arange(subLag[0], subLag[-1], 0.01)
    
    mean = sum(subLag * subCorr) / sum(subCorr)
    sigma = np.sqrt(abs(sum(subLag * (subCorr - mean) ** 2) / sum(subCorr)))
    
    if sigma > 1000: # this improves some of the errors caused by oversized sigma
        np.multiply(subCorr, 100)
        mean = sum(subLag * subCorr) / sum(subCorr)
        sigma = np.sqrt(abs(sum(subLag * (subCorr - mean) ** 2) / sum(subCorr)))
    
    # this can be pretty janky with edges and shit
    with warnings.catch_warnings():
        try:
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(Gauss, subLag, subCorr, p0=[max(subCorr), mean, sigma])
            #popt, _ = curve_fit(Gauss, subLag, subCorr, p0=[max(subCorr), mean])
            timeLagGauss = newX[np.argmax(Gauss(newX, *popt))]
        except:
            timeLagGauss = timeLag
    
    return(maxCorr, timeLag, timeLagGauss)


def fullCorrOnePointV3(inputData, testPointIndex):
    
    '''
    Does the fullCorrOnePoint job, but in a less stupid way.
    
    Given inputData (a (t,x) DTS array) and testPointIndex, correlate the
    test point against all other points.
    
    --> skips extraneous data from fullCorrOnePoint()
    --> automatically limits negative distances (redundant info)
    
    Part of fullCorrOneHeight()
    
    NB: there is a V2 which includes a correlation coefficient threshold in
        BLM_Processing.py if you need it. There is ALSO a V2 of fullCorrOneHeight
        which implements similar limits but probably shouldn't be used. If
        you want to limit distance chop the input array instead.
    
    Parameters
    -----------
    inputData: a (t,x) DTS temp np array
    testPointIndex: integer (0:x), the index of the test point
    
    Returns
    -------
    A dictionary with the following keys/ordered lists:
        corrCoeff - the max correlation coefficients
        distance_m - the distance in metres between the time series
        timeLag_idx - time lag in indices of max correlation
        timeLag_Gauss - time lag in indices of max correlation (gauss fit)
    '''
    
    refTimeSeries = inputData[:,testPointIndex]
    
    # data holders
    corrCoeffList = []
    timeLagList = []
    timeLagGaussList = []
    distanceList = []
    
    a = datetime.now()
    #bigA = a
    
    for idx in range(inputData.shape[1]):
        
        testTimeSeries = inputData[:,idx]
        if np.isnan(testTimeSeries[0]): # skip masked values
            continue
        distance_idx = idx - testPointIndex
        if distance_idx < 0: # skip negative (redundant) distaances
            continue
        
        
        corrCoeff, timeLag_idx, timeLag_idxGauss = singleLag(refTimeSeries, testTimeSeries)
        distance_metres = distance_idx * 0.254
        
        corrCoeffList.append(corrCoeff)
        timeLagList.append(timeLag_idx)
        distanceList.append(distance_metres)
        timeLagGaussList.append(timeLag_idxGauss)
        #print(len(distanceList))
    
    '''
    # this is a countdown
    if (x:=len(distanceList)) % 25 == 0: 
        print(x, end = ' ... ')
        minutes, seconds = divmod((datetime.now() - a).seconds * 25, 60)
        print('%i min %i s' % (minutes, seconds))
        #a = datetime.now()
    '''
    
    return({'corrCoeff':corrCoeffList, 'timeLag_idx':timeLagList,
            'distance_m':distanceList, 'timeLag_Gauss':timeLagGaussList})


def Gauss(x, a, x0, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return(a * np.exp(-(x - x0)**2 / (2 * sigma**2)))


def fullCorrOneHeight(inData, collapse=True, distance=40, symmetry=False):
    
    '''
    Does the correlation routine for an entire height of DTS data by iterating
    over each point and running fullcorronepoint().
    
    Parameters
    ----------
    inData: (t,x) DTS data. Should be masked.
    collapse: T/F - collapses the dictionaries in to more useful form
    OLD - distance: integer, metres, limits the correlation to only +/-
                a given distance
    OLD - symmetry: T/F calculate all distances or only positives?
    
    Returns
    ---------
    Either a bulky dictionary or a collapsed dictionary
    '''
    
    if len(inData.shape) > 2:
        raise ValueError("input data should only be two dims, yo")
        
    a = datetime.now()
    bigA = a
    
    print(" * Total: %i" % inData.shape[1])
        
    bigDictionary = {}
    if distance is not None: # legacy
        limitDistance=True
    else:
        limitDistance=False
        
    for idx, point in enumerate(range(inData.shape[1])):
        #if point % 50 == 0:print(point, end=' ... ')
        
        if np.isnan(inData[0,point]):
            continue

        key = str(point).zfill(4) 
        # old version
        '''
        bigDictionary[key] = fullCorrOnePoint(inData, point, 
                                              limitDistance=limitDistance,
                                              distance=distance,
                                              symmetry=symmetry)
        '''
        
        bigDictionary[key] = fullCorrOnePointV3(inData, point)
        
        if idx % 25 == 0 and idx != 0:
            m, s = divmod((datetime.now() - a).seconds, 60)
            print('%i | %i m %i s' % (idx, m, s))
            
    m, s = divmod((datetime.now() - bigA).seconds, 60)
    print(" * Total: %i m %i s" % (m, s))
            
    if collapse:
        print('collapsing dictionaries')
        #bigDKeys = bigDictionary.keys()
        # this is fuck ugly
        keyOne = list(bigDictionary.keys())[0]
        flatDictionary = {key:[] for key in bigDictionary[keyOne]}
        
        for key in bigDictionary:
            subDat = bigDictionary[key]
            
            for subKey in subDat:
                flatDictionary[subKey].extend(subDat[subKey])
                
        return(flatDictionary)
    else:
        return(bigDictionary)
    

def singleLag_testing(ts1, ts2):
    
    '''Does the single lag for two time series and returns the normalised
    autocorr function'''
        
    ts1_m = signal.detrend(ts1)
    ts2_m = signal.detrend(ts2)
    #denom = np.sqrt(np.std(ts1_m) * np.std(ts2_m))

    #nx = len(ts1)
    #lags = np.arange(-nx+1, nx)
    
    autocorr = np.correlate(ts1_m, ts2_m, mode='full')
    autocorr *= 1/len(ts1)
    autocorr /= np.sqrt(np.var(ts1_m) * np.var(ts2_m))    
    
    return(autocorr)


def avgSegment(data, filter_val):
    
    '''
    In this context used to average the output of the full correlation
    routine. Separates 'data' in to bins determined by 'filter_val' and then
    averages everything within each bin.
    
    If a = one of the flattened output dictionaries, use something like
    filter, vals = avgSegment(a['timeLag_s'], a['distance_m'])
    
    In principle it's more flexible than that but w/e
    '''
    
    filter_vals = np.unique(filter_val)
    averaged_vals = []
    
    for idx, x in enumerate(filter_vals):
        if idx % 25 == 0: print(round(x, 2), end=' ... ')
        dat_filter = (filter_val == x)
        avg = np.nanmean(list(compress(data, dat_filter)))
        averaged_vals.append(avg)
        
    return(filter_vals, averaged_vals)


def weightedAvgSegment(data, filter_val, weight_val, lims=True):
    
    '''
    Does the same as the avgSegment() function but also allows you to 
    specify a weight_val variable. This will be used to weight the average 
    within each bin. So far I've used it to weight the average by the
    correlation coefficient.
    
    lims: T/F returns the standard deviation within each bin, useful for 
        certain plots/examining the spread w/i each bin.
    '''
    
    filter_vals = np.unique(filter_val)
    averaged_vals = []
    if lims: sd = [] # track standard dev. of each bin
    
    print(" -> weighted averaging")
    print(" -> going to %.2f" % filter_vals[-1])
    for idx, x in enumerate(filter_vals):
        if idx % 50 == 0: print(round(x, 2), end=' ... \n')
        dat_filter = (filter_val == x)
        
        val_sub = np.array(list((compress(data, dat_filter))))
        weight_sub = np.array(list(compress(weight_val, dat_filter)))
        
        if lims: sd.append(np.std(val_sub))
        
        averaged_vals.append(np.sum(val_sub * weight_sub) / np.sum(weight_sub))
        
    if lims: return(filter_vals, averaged_vals, sd)
    return(filter_vals, averaged_vals)
    

def getCorrLimit(sd, threshold=1):
    
    '''
    given the SD of individual bins this returns the index at which the sd
    starts to rapidly increase (i.e. sd[i+1]-sd[i] > threshold)
    
    NB: the index that gets returned is python indexed, so series[:limit] will
    give the desired result
    '''
    
    sdnew = []
    for i in range(1, len(sd)):
        sdnew.append(abs(sd[i]-sd[i-1]))
        
    over = np.where(np.array(sdnew)>threshold)[0]
    limit = over[1] if over[0] == 0 else over[0]
    
    return(limit)


def convVelocity(data, verbose=True, fullOut=False):
    
    '''
    -> calculate the convection velocity given some input data.
    -> data should be 2d (t,x) and already cropped to the times you want
        -> if you later incorporate both axes of the array you need to run
            this twice and/or modify
    '''
    
    if (x := len(data.shape)) > 2:
        raise ValueError("Currently only 2d (t,x) data is accepted: %i" % x)
        
    # step 1 -> do the correlation
    if verbose: print(' * correlating')
    a = fullCorrOneHeight(data) 
    # no limit distance at the mo - lots of wasted processing there...?
    
    # step 2 -> avg lags w/i each distance bin
    if verbose: print(' * averaging')
    dist, avglag_idx, sd = weightedAvgSegment(a['timeLag_Gauss'], a['distance_m'],
                                              a['corrCoeff'], lims=True)
    
    # when does the relationship fail?
    stopI = getCorrLimit(sd, threshold=1)
    
    if stopI < 3: 
        print("\n ** NO RELATIONSHIP: 0 m/s")
        return(0) # TODO: what is a good threshold?
    
    distC = np.array(dist[1:stopI])
    avglag_s = np.array([x*1.288 for x in avglag_idx[1:stopI]])
    
    velocities = np.divide(distC, avglag_s)
    
    # obvious outliers? (clip vals > 3*sd)
    vel_sd = np.std(velocities)
    vel_mean = np.mean(velocities)
    tester = np.array([abs(x-vel_mean) for x in velocities])
    valids = velocities[tester < 3 * vel_sd]
    outliers = velocities[tester > 3 * vel_sd]
    
    conv_vel = np.mean(valids)
    
    if verbose: 
        print('\n * -- Convective velocity results:')
        print('      measured velocity: %.2f m/s' % conv_vel)
        print('      over %.2f metres, dropped %i outlier(s)' % 
              (stopI*.254, len(outliers)))
    
    if fullOut:
        return(dist, avglag_idx, sd, stopI, velocities, valids)
    else:
        return(conv_vel)
    
    


# --- Plotting Helpers


def corrCoeffPlot(dset, fit, title=None, m=None):
    
    '''
    Formerly called 'thatPlot()', this plots 'that plot'. i.e. the fitted 
    correlation function at a set of given distances.
    
    Input
    ------
    dset: (t,x) DTS dataset
    fit: T/F apply a polynomial fit or keep raw? Raw is super noisy.
    title: optional, title for the figure
    m: optional, a series of metre-distances to plot the function for
    '''
        
    t1 = dset[:,5]
    tries = range(5, 594, 1)
    dic = {}
    lags = np.arange(-len(t1)+1, len(t1))
    
    try:
        start = list(lags).index(-120)
        end= list(lags).index(121)
    except:
        start = 0
        end = len(lags)

    for step in tries:
        #print(step)
        if np.isnan(dset[:,step][0]):
            continue
        corr = singleLag_testing(dset[:,5], dset[:,step])
        dic[str(step).zfill(3)] = corr
    
    ids = list(dic.keys())

    if m is None:
        m = [1, 4, 8, 16, 32, 64, 128]
    
    seq = [int(x / .254) + 1 for x in m]
    plt.figure(figsize=(16,12))
    count = 2
    for idx_dist in seq:
        try:
            key = ids[idx_dist]
        except:
            break
        if count % 2 == 0:
            l = '-'
        else:
            l = '--'
        
        count += 1
        spatialDist = (int(key)-5) * 0.254
    
        if fit:
            y = dic[key][start:end]
            z = np.polyfit(lags,y,31)
            p = np.poly1d(z)
            plt.plot(lags*1.288, p(lags), l, label=str(idx_dist)+' <- idx | metres -> '+str(round(spatialDist, 2)))
        else:
            plt.plot(lags*1.288, dic[key], l, label=str(idx_dist)+' <- | metres -> '+str(round(spatialDist, 2)))
   
    plt.xlim(-120, 120)
    plt.ylim(-0.5, 0.7)
    plt.legend()
    plt.xlabel('time lag (s)')
    plt.ylabel('correlation coeff [-1, 1]')
    if title is not None:
        plt.title(title)
        
        
def bin_ids(n, bins):
    
    '''gets the IDs that should be used for binning distributions. To do this
    right the bins should be logarithmic'''
    
    return(np.arange(0, n, n//bins))


def binAvgThisShit(binIDs, dat):
    
    '''bin averages this shit, using the binIDs from bin_ids()'''
    
    avgs = []
    for i in range(0, len(binIDs)-1):
        avgs.append(np.nanmean(dat[binIDs[i]:binIDs[i+1]]))
    return(avgs)


def plotCoSpec(spec, nsmooth=1, title=None, save=None):
    
    '''
    Plots the co-spectrum and phase spectrum output by the pycurrents
    function.
    
    Input
    ------
    spec: a pycurrents.spectrum object
    nsmooth: what nsmooth did you use on the original spectrum calc?
    title: optional, what title for hte plot?
    save: optional, filename defining where to save the plot
    '''
    
    fig, ax = plt.subplots(nrows=3, sharex=True)
    if title is not None:
        fig.suptitle(title)
    ax[0].loglog(spec.freqs, spec.psd_x, 'b',
                 spec.freqs, spec.psd_y, 'r')
    ax[0].set_ylabel("PSD")
    #ax[0].set_ylim(np.min(spec.psd_x), np.max(spec.psd_x))
    
    ax[1].semilogx(spec.freqs, spec.cohsq)
    ax[1].set_ylabel("coh$^2$")
    ax[1].set_ylim(0,1)
    #c95 = np.nan if nsmooth <= 1 else 1 - 0.05 ** (1 / (nsmooth-1))
    #ax[1].axhline(c95, color='pink')
    
    ax[2].semilogx(spec.freqs, np.rad2deg(spec.phase), 'o', markersize=3)
    ax[2].set_ylim(-180, 180)
    ax[2].set_ylabel('phase')
    #ax[2].set_xlim(spec.freqs[nsmooth], spec.freqs[-nsmooth])
    ax[2].set_yticks([-90, 0, 90])
    ax[2].set_xlabel('Freq')
    
    #if title is not None:
    #    ax[0].title(title)
        
    if save is not None:
        plt.savefig(save)
        
        
def plotAvgAndSD(dist, avg, sd, save=None):
    
    '''
    I think this plot could be useful but at the moment this function is 
    very inflexible. Adapt later as necessary
    '''
    
    y1 = np.add(avg, sd)
    y2 = np.add(avg, [-x for x in sd])
    
    fig,ax=plt.subplots()
    ax.plot(dist, avg)
    ax.fill_between(dist, y2, y1, alpha=0.1)
    
    plt.xlabel('distance (m)')
    plt.ylabel('time lag (idx)')
    plt.title('dist mean +/- sd')
    
    if save is not None:
        plt.savefig(save)
        

def gifFolder(folder, saveFolder, savename, ext='*.png', duration=0.1):
    
    imNames = glob.glob(os.path.join(folder, ext))
    imNames.sort()
    images = []
    for fname in imNames:
        images.append(imageio.imread(fname))
        
    print("* Saving gif of %i images" % len(images))
    imageio.mimsave(os.path.join(saveFolder, savename), images, duration=duration)
        

def FreqPlots3d(dataFrame, ppf, norm=None, saveFolder=None, freqLim=20,
                dpi=250, forceAz=None):
    
    '''
    Just wraps some of the common 3d plotting I've been doing in a function,
    aiming for flexibility, ending in over-complexity
    
    Parameters
    ----------
    dataFrame: a pd dataframe with the avgCohsqTShift output
    ppf: how many plots per individual frame. Atm can be 1 or 2
    norm: if only 1 plot per frame, this needs to be defined. Is the z-axis
        1 = linear, normalised to [0,1] for each frequency/frame, or
        2 = log10, normalised across the whole dataset.
        -> this could be made more flexible
    saveFolder: optional. If a folder is given, the plots are saved there and
        numbered sequentially. if None, plots are displayed
    freqLim: integer, how many frequencies to iter over / plots to create
    dpi: plot dpi.
    '''
    
    cm = plt.get_cmap('plasma')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    freqs = dataFrame['freq'].unique()
    
    if ppf not in [1,2]: raise ValueError('plots per frame must be 1 or 2 :/')
    if ppf == 1 and norm is None: raise ValueError('norm must be defined if ppf is 1')
    
    print(' ** plotting')
    print(' %i plots per frame' % ppf)
    if saveFolder is not None:
        print(' -> saving output plots to %s' % saveFolder)
    else:
        print(' -> NOT saving output')
    
    if ppf == 1:
        for idx, f in enumerate(freqs[:freqLim]):
            
            Xs = dataFrame['dist_m'][dataFrame['freq']==f]
            Ys = dataFrame['time_s'][dataFrame['freq']==f]
            Zs = dataFrame['coh'][dataFrame['freq']==f]
            
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            if norm:
                Zrange = Zs.max()-Zs.min()
                Znorm = (Zs-Zs.min()) / Zrange
                ax.scatter(Xs,Ys,Znorm, c=scalarMap.to_rgba(Znorm), alpha=0.25)
                ax.set_zlim(0,1)
                ax.set_zlabel('norm. linear coh')
            else:
                ax.scatter(Xs,Ys,np.log10(Zs), c=scalarMap.to_rgba(Zs), alpha=0.25)
                ax.set_zlim(-1,0.)
                ax.set_zlabel('log10(coh)')

            ax.set_zticks([])
            ax.set_ylabel('time lag(s)')
            ax.set_xlabel('distance (m)')
            
            if forceAz is not None: ax.azim=forceAz
            
            title = 'P: ' + str(round(1/f, 2)) + ' s | f: ' + str(round(f, 3)) + ' Hz'
            plt.title(title)
            
            if saveFolder is not None:
                plt.savefig(os.path.join(saveFolder, str(idx).zfill(4) + '.png'))
            
    elif ppf == 2:
        for idx, f in enumerate(freqs[:freqLim]):
            
            Xs = dataFrame['dist_m'][dataFrame['freq']==f]
            Ys = dataFrame['time_s'][dataFrame['freq']==f]
            Zs = dataFrame['coh'][dataFrame['freq']==f]
    
            Zrange = Zs.max()-Zs.min()
            Znorm = (Zs-Zs.min()) / Zrange
            
            fig=plt.figure(dpi=dpi, figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(Xs,Ys,Znorm, c=scalarMap.to_rgba(Znorm), alpha=0.25)
            ax.set_zlim(0,1)
            ax.set_xlabel('distance (m)')
            ax.set_ylabel('time lag (s)')
            ax.set_zlabel('linear coh')
            ax.set_zticks([])
            plt.title('normalised')
            
            if forceAz is not None: ax.azim=forceAz
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(Xs,Ys,np.log10(Zs), c=scalarMap.to_rgba(Zs), alpha=0.25)
            ax2.set_zlim(-1,0.)
            ax2.set_xlabel('distance (m)')
            ax2.set_ylabel('time lag (s)')
            ax2.set_zlabel('log10(coh)')
            ax2.set_zticks([])
            plt.title('non-normalised')
            
            if forceAz is not None: ax2.azim=forceAz
    
            title = 'P: ' + str(round(1/f, 2)) + ' s | f: ' + str(round(f, 3)) + ' Hz'
            plt.suptitle(title)       
            
            if saveFolder is not None:
                plt.savefig(os.path.join(saveFolder, str(idx).zfill(4) + '.png'))


def limCheck(dataframe, distances, freqs, lags):
    
    print(' ** limits :(')
    lagPeaks = []
    for f in freqs:
        for d in distances:
            cSlice = dataframe['coh'][(dataframe['dist_m']==d) &
                                  (dataframe['freq']==f)]
            lagPeaks.append(lags[np.where(cSlice==cSlice.max())[0]])
    
    print(np.min(lagPeaks), np.max(lagPeaks))
    return(np.min(lagPeaks), np.max(lagPeaks))


def wavelengthVelPlot1(dataframe, distances, f, lags, idx=None, lims=None,
                       line=False, save=False):
    
    '''
    Explain yourself
    '''
    
    lagPeaks = []
    colours = []
    
    fig = plt.figure(dpi=250, figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(121)
    
    for d in distances:
        col = next(ax._get_lines.prop_cycler)['color']
        colours.append(col)
        
        cSlice = dataframe['coh'][(dataframe['dist_m']==d) &
                                  (dataframe['freq']==f)]
        lag = lags[np.where(cSlice==cSlice.max())[0]]

        plt.plot(lags, cSlice, color=col)
        plt.plot(lag, cSlice.max(), 'o', color=col)
        plt.ylim(0,1)
        plt.ylabel('coh')
        plt.xlabel('time lag (s)')
        
        lagPeaks.append(lag)
    plt.title('P: ' + str(round(1/f, 2)) + ' s | f: ' + str(round(f,3)) + ' Hz')

    ax2 = fig.add_subplot(122)
    #plt.gca().set_prop_cycle(None)
    
    for idxD, d in enumerate(distances):
        #col = next(ax._get_lines.prop_cycler)['color']
        plt.plot(d, lagPeaks[idxD], 'o', color=colours[idxD])
    
    plt.ylabel('peak lag (s)')
    plt.xlabel('distance (m)')
    if lims is not None:
        l, h = lims
        plt.ylim(l, h)
    
    if line:
        lagPeaks = np.array([item for sublist in lagPeaks for item in sublist])
        maskP = lagPeaks[lagPeaks!=0]
        maskD = distances[lagPeaks!=0]
        
        calcVels = np.divide(maskD, maskP)
        m = np.mean(calcVels)
        sd = np.std(calcVels)
        
        plt.title('mean: %.2f m/s | sd: %.2f' % (m, sd))
        
        l = linregress(distances, lagPeaks)
        plt.plot(distances, [l.slope * x for x in distances],
                 'k-',label=str(round(l.rvalue**2, 2)))
        plt.legend()
        
    if save:
        #print('saving')
        if not isinstance(save, str): raise ValueError("y'fucked up")
        if save and idx is None: raise ValueError("gimme an index")
        #print(idx)
        plt.savefig(os.path.join(save, str(idx).zfill(4)+'.png'))
        
        
def wavelengthVelPlot2(dataframe, peakFrame, distances, f, lags, idx=None, 
                       lims=None, line=False, save=False):
    
    '''
    Explain yourself
    '''
    
    lagPeaks = []
    colours = []
    
    fig = plt.figure(dpi=250, figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(121)
    
    for d in distances:
        col = next(ax._get_lines.prop_cycler)['color']
        colours.append(col)
        
        cSlice = dataframe['coh'][(dataframe['dist_m']==d) &
                                  (dataframe['freq']==f)]
        #lag = lags[np.where(cSlice==cSlice.max())[0]]
        peakLag = peakFrame['peakLag_s'][(peakFrame['dist_m']==d) &
                                         (peakFrame['freq']==f)]
        peakCoh = peakFrame['peakCoh'][(peakFrame['dist_m']==d) &
                                       (peakFrame['freq']==f)]

        plt.plot(lags, cSlice, color=col)
        #plt.plot(lag, cSlice.max(), 'o', color=col)
        plt.plot(peakLag, peakCoh, 'o', color=col)
        plt.ylim(0,1)
        plt.ylabel('coh')
        plt.xlabel('time lag (s)')
        
        lagPeaks.append(peakLag)
    plt.title('P: ' + str(round(1/f, 2)) + ' s | f: ' + str(round(f,3)) + ' Hz')

    ax2 = fig.add_subplot(122)
    #plt.gca().set_prop_cycle(None)
    
    for idxD, d in enumerate(distances):
        #col = next(ax._get_lines.prop_cycler)['color']
        plt.plot(lagPeaks[idxD], d, 'o', color=colours[idxD])
    
    plt.xlabel('peak lag (s)')
    plt.ylabel('distance (m)')
    if lims is not None:
        l, h = lims
        plt.xlim(l, h)
        #plt.xlim(0, distances[-1])
    
    if line:
        lagPeaks = np.array([item for sublist in lagPeaks for item in sublist])
        maskP = lagPeaks[lagPeaks>0.01]
        maskD = distances[lagPeaks>0.01]
        
        calcVels = np.divide(maskD, maskP)
        m = np.mean(calcVels)
        sd = np.std(calcVels)
        
        plt.title('mean: %.2f m/s | sd: %.2f' % (m, sd))
        
        l = linregress(maskD, maskP)
        plt.plot([l.slope * x for x in distances], distances,
                 'k-',label=str(round(l.rvalue**2, 2)))
        plt.legend()
        
    if save:
        #print('saving')
        if not isinstance(save, str): raise ValueError("y'fucked up")
        if save and idx is None: raise ValueError("gimme an index")
        #print(idx)
        plt.savefig(os.path.join(save, str(idx).zfill(4)+'.png'))

# --- Spectral work

def discreteEnergySpectrum(ts, timeOrSpace):
    
    ''' 
    Calculates the discrete variance/energy spectrum as per Stull 88 p.313
    for a single transect.
    
    NB: I may be doing this not-correctly. Better to rely on someone else's 
    work imo.

    Inputs
    ------
    ts: a time/space series. 1-Dimensional.
    timeOrSpace: string, 'time' or 'space'. In which domain to calculate? This
        is important for defining the output frequencies.
        
    Returns
    -------
    freqs: a list of frequencies over which the FFT is calculated
    dESpectrum: the discrete variance intensity at a given frequency
    N: size of the ts
    '''
    
    if np.nanmean(ts) > 1: 
        ts = signal.detrend(ts)
        
    if timeOrSpace == 'time':
        freqs = fftfreq(ts.size, d=1/1.288)
    elif timeOrSpace == 'space':
        freqs = fftfreq(ts.size, d=0.254)
    N = ts.size
    #freqs = fftfreq(ts.size, d=1/1.288) # sample rate is hardcoded here
    freqs = freqs[1:N//2]
    f_ts = fft(ts)
    
    if N % 2 == 0: # even
        #nf = N//2
        dESpectrum = [2*np.abs(x)**2 for x in f_ts[1:N//2]]
        #dESpectrum.append(f_ts[N//2]**2)
    else: # odd
        dESpectrum = [2*np.abs(x)**2 for x in f_ts[1:N//2]]
    
    return(freqs, dESpectrum, N)


def avgCohsqTShift(data, shifts, nsmooth, collapse, limitdistance, nfft,
                   dFrameOut):
    
    '''
    Average Coherence Squared + Time Shift
    
    Here we calculate the coherence spectrum (though could be modified to
    handle a wider variety of spectra) over an entire height PLUS with time
    shifts. I.e. from every point TO every point, plus with time shifts,
    and then aggregate all the results.
    
    Parameters
    ----------
    data: a (t,x) DTS transect
    shifts: the range of time shifts (in indices) over which to iterate, e.g.
        np.arange(-ts, ts+1)
    nsmooth: parameter of the spectrum calculation, should be odd number
    collapse: T/F - collapse final dictionary in to a more useful form?
    limitdistance: distance in indices to cut off (if you want that)
    nfft: parameter of the spectrum calculation. Should stay None I think.
    dFrameOut: whether or not to convert the output in to a data frame. Makes
        the data useful, but is very time consuming (and probably inefficient)
        
    Output
    ------
    Can be a few things, see inputs and return statements.
    
    Collapsed dictionary has the form dict[distance_idx][timelag_idx] and
    each of those subsequent arrays containes per-frequency coherence squared
    and has dimensions (frequencies, n) where n is however many overlapping
    distances and timelags there are.
    
    If you dFrameOut then the output is a (large) 'tidy' pandas data frame with
    column names dist_m, time_s, freq, coh
    
    --> if you do ANY intermediary averaging in time or space you'll need to
        go back and look at this stuff, all conversion factors will need to be
        adjusted
    '''
    
    bigD = {}
    
    # do we need this?
    dimTest = spectra.spectrum(data[:,5], dt=1/1.288, nfft=nfft, smooth=nsmooth)
    outFreqs = dimTest.freqs
    
    a = datetime.now()
    bigA = a
    print(" * TODO: %i" % data.shape[1])
    for outercounter, point in enumerate(range(data.shape[1])):
        if outercounter % 5 == 0 and outercounter != 0: 
            validP = 0
            print(outercounter, end=' ... ')
            print(str((datetime.now() - a).seconds) + ' s', end=' | ')
            print(validP)
            a = datetime.now()
        
        refSeries = data[:,point] # go through each input point
        if np.isnan(refSeries[0]): continue # if array is masked
        
        for innercounter, tpoint in enumerate(range(data.shape[1])):
            #print(innercounter)
            dist_idx = tpoint - point
            if dist_idx <= 0: continue # no coherence against self, negatives redundant
            if limitdistance is not None and dist_idx > limitdistance: continue
            
            testSeries = data[:,tpoint]
            if np.isnan(testSeries[0]): 
                #print('wow')
                continue
        
            validP += 1
            
            distKey = str(dist_idx).zfill(4)
            if distKey not in bigD:
                bigD[distKey] = {}
            
            for shiftcounter, shift in enumerate(shifts):
                
                nFill = 3 if shift > 0 else 4
                shiftKey = str(shift).zfill(nFill) # increase if you go over 999!!
                newTS = shiftSignal(testSeries, shift, fill_value=0)
                spec = spectra.spectrum(refSeries, newTS, dt=1/1.288, nfft=None, smooth=nsmooth)
                
                if shiftKey not in bigD[distKey]: # on the first runthrough establish the dic keys
                    #bigD[shiftKey] = spec.cohsq[..., np.newaxis] # SUPER SLOW
                    bigD[distKey][shiftKey] = [list(spec.cohsq)]
                else:
                    #bigD[shiftKey] = np.append(bigD[shiftKey], spec.cohsq[..., np.newaxis], axis=1)
                    bigD[distKey][shiftKey].append(list(spec.cohsq))
                    
    elapsed = str(datetime.now() - bigA)
    
    if collapse:
        print('collapsing ... ')
        for key1 in bigD:
            print(key1, end=' ')
            for key2 in bigD[key1]:
                bigD[key1][key2] = np.mean(np.swapaxes(np.array(bigD[key1][key2]), 0, 1), axis=1)
    
    print('done ' + elapsed)
    
    # convert to dataframe if you want. Probably always want to.
    if dFrameOut: return(cohSqDF(outFreqs, bigD))

    return(outFreqs, bigD)        


def avgCohsqTShift_noPadding(data, maxTimeLag_s, limitDist_or_idxList, 
                             nsmooth, nfft, collapse, dFrameOut, updateIter=10,
                             extractPeaksOnly=False):
    
    '''
    Average Coherence Squared + Time Shift
     ** No padding version **
    
    Here we calculate the coherence spectrum (though could be modified to
    handle a wider variety of spectra) over an entire height PLUS with time
    shifts. I.e. from every point TO every point, plus with time shifts,
    and then aggregate all the results.
    
    ** 2020.11.30 --> adding the extractPeaksOnly parameter. Saving the 
        complete curves IS useful, especially for visualisation, but I think
        the memory usage required eats resources. extractPeaksOnly=True will
        only save a gaussian fit of the peak and reduce the memory usage by
        a pretty significant factor
        ** --> Wait actually I'm not 100% how to do this intelligently rn...
        --> I don't know if there is actually a smart way to do this faster as
            we need to average all the iterations first to derive the function
            
    ** 2020.12.01 --> modified the innermost loop so that 
    
    Parameters
    ----------
    data: a (t,x) DTS transect
    maxTimeLag_s: the max time lag to evaluate, in seconds. E.g. if you give
        max time lag of 10 seconds, lags from -10s to +10s will be calculated.
        Must be an integer, NOT a list of lags like the previous version
    limitDist_or_idxList: None, int, or list.
        None - will calculate over every other +ve point
        Int - distance to stop at in metres. Will calculate up to this distance,
            but no further
        List - a list of indices to evaluate. Will ONLY calculate coherence at
            those specific distances
    nsmooth: parameter of the spectrum calculation, should be odd number
    nfft: parameter of the spectrum calculation. Should stay None I think.
    collapse: T/F - collapse final dictionary in to a more useful form?
    dFrameOut: whether or not to convert the output in to a data frame. Makes
        the data useful, but is very time consuming (and probably inefficient)
    updateIter: how often to give updates. Default is every 10 points.
    extractPeaksOnly: if True, will only save the gaussian peak time lag for
        each (frequency, distance) subset
        
    Output
    ------
    Can be a few things, see inputs and return statements.
    
    Collapsed dictionary has the form dict[distance_idx][timelag_idx] and
    each of those subsequent arrays containes per-frequency coherence squared
    and has dimensions (frequencies, n) where n is however many overlapping
    distances and timelags there are.
    
    If you dFrameOut then the output is a (large) 'tidy' pandas data frame with
    column names dist_m, time_s, freq, coh
    
    --> if you do ANY intermediary averaging in time or space you'll need to
        go back and look at this stuff, all conversion factors will need to be
        adjusted
    --> NB: possible improvement would be to store the sums within each
        individual key as well as an iteration tracker, could get the mean
        from that w/ much less memory usage?
    '''
    
    bigD = {}
    validP = 0
    
    # time shift preamble
    maxTimeLagIdx = int(maxTimeLag_s//1.288)
    totalShifts = maxTimeLagIdx * 2 + 1 # *2 covers negatives, +1 is 0-lag
    lags_idx = np.arange(-maxTimeLagIdx, maxTimeLagIdx+1) # I think this is right
    
    # store frequencies now for output, these don't change
    # nb: this is only valid for time spectra, need to change some constants
    #     to apply it to space
    dimTest = spectra.spectrum(data[maxTimeLagIdx:-maxTimeLagIdx-1,5], dt=1/1.288,
                               nfft=nfft, smooth=nsmooth)
    outFreqs = dimTest.freqs
    
    # distance preamble
    if limitDist_or_idxList is None:
        limMethod = None
        limitDistance_m = data.shape[1] * 0.254
    elif isinstance(limitDist_or_idxList, int):
        limMethod = 'dist'
        limitDistance_m = limitDist_or_idxList
        limitDistance_idx = int(limitDistance_m/0.254)
    elif isinstance(limitDist_or_idxList, list):
        limMethod = 'list'
        validIdxList = limitDist_or_idxList
        limitDistance_m = validIdxList[-1] * 0.254
    else:
        raise TypeError("invalid limit: None, Int, List (see docstring)")
        
    nDistances = len(limitDist_or_idxList) # this only works if you're using the list
    nPoints = data.shape[1]
    
    nIter = (nPoints * nDistances * totalShifts)
    nSeconds = ((nPoints * 0.001136) + (nDistances * nPoints * 0.0004) + 
                (nDistances * nPoints * totalShifts * 0.00092))
    minutes, seconds = divmod(nSeconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(' * ---- Coherence Spectrum')
    print('   -> +/- %i seconds (%i total indices)' % (maxTimeLag_s, lags_idx.size))
    print('   -> over %.2f metres' % limitDistance_m)
    print('   -> Total indices to do: %i' % data.shape[1])
    print('   -> Forecast: %i iterations | %i hours, %i minutes, %i seconds' %
          (nIter, hours, minutes, seconds))
    print(' * ---- STARTING WOOHOO')
    
    a = datetime.now()
    bigA = a
    iCounter = 0
    
    for outercounter, point in enumerate(range(data.shape[1])):
        
        if outercounter % updateIter == 0 and outercounter != 0:
            print(outercounter, end=' ... ')
            m, s = divmod((datetime.now() - a).seconds, 60)
            print('%i min %i s | ' % (m, s), end='')
            print('~%i points, %i iterations' % (int(validP/updateIter), 
                                                 int(validP*totalShifts)))
            validP = 0
            a = datetime.now()
            
        # going through each point in space. Reference series is cropped
        # to a more limited range, and the 'test' series is repeatedly shifted
        refSeries = data[maxTimeLagIdx:-maxTimeLagIdx-1, point]
        
        if np.isnan(refSeries[0]): continue # skip if the point is masked
        
        # I think we want to detrend first due to the full series shifting
        refSeries = signal.detrend(refSeries) 
        
        for innercounter, tpoint in enumerate(range(data.shape[1])):
            
            dist_idx = tpoint - point
            if dist_idx <= 0: continue # skip redundant information / autocorr
            
            if limMethod == 'dist' and dist_idx > limitDistance_idx:
                continue # skip if > limit distance and lim dist. is an integer
                
            if limMethod == 'list' and dist_idx not in validIdxList:
                continue # skip if not in the defined id list
                
            testSeries = data[:, tpoint]
            if np.isnan(testSeries[0]): continue # skip masked points
            validP += 1
            
            # I thin kwe want to detrend first due to the full series shift
            testSeries = signal.detrend(testSeries)
            
            distKey = str(dist_idx).zfill(4) # first set of keys for bigD
            if distKey not in bigD:
                bigD[distKey] = {} # initialise empty dictionary
                
            # this is the big mod to the previous version. Instead of using
            # the shiftSignal function to 0-pad, we selectively crop the 
            # full 30-min time series
            for idx in range(totalShifts + 1): # this is fuck ugly
                
                if idx == totalShifts: continue
                lag_idx = lags_idx[idx]
                shiftKey = str(-lag_idx).zfill(4) # valid up to 9999
            
                startCropI = idx
                endCropI = -maxTimeLagIdx * 2 + (idx-1) # VERY fuck ugly
                
                if endCropI < 0:
                    testSeries_shift = testSeries[startCropI:endCropI]
                else:
                    testSeries_shift = testSeries[startCropI:]
            
                spec = spectra.spectrum(refSeries, testSeries_shift, dt=1/1.288,
                                        nfft=nfft, smooth=nsmooth)
                
                iCounter += 1
                
                if len(spec.cohsq) != len(outFreqs):
                    raise ValueError("Lengths don't match? Point/TPoint/lag_idx:", 
                                     point, tpoint, lag_idx)
                
                ''' 2020.12.01 modification. Uses less memory and is WAY faster in collapse
                if shiftKey not in bigD[distKey]: # initialise list
                    bigD[distKey][shiftKey] = [list(spec.cohsq)]
                else:                             # populate list
                    bigD[distKey][shiftKey].append(list(spec.cohsq))
                '''
                
                if shiftKey not in bigD[distKey]:
                    bigD[distKey][shiftKey] = [0,spec.cohsq]
                else:
                    bigD[distKey][shiftKey][0] += 1
                    bigD[distKey][shiftKey][1] = np.add(bigD[distKey][shiftKey][1], spec.cohsq)
            
    elapsed = str(datetime.now() - bigA)
    print('done in ' + elapsed, end='')
    print(' | versus forecast: %i:%i:%i' % (hours, minutes, seconds))
    print('estimated iterations: %i' % nIter)
    print('actual iterations (spectra calculated): %i' % iCounter)
    
    if collapse:
        print('collapsing ... ')
        for key1 in bigD:
            print(key1)
            for key2 in bigD[key1]:
                #bigD[key1][key2] = np.mean(np.swapaxes(np.array(bigD[key1][key2]), 0, 1), axis=1)
                bigD[key1][key2] = np.divide(bigD[key1][key2][1], bigD[key1][key2][0])
    
    # optional: convert to data frame. Probably always want to do this.
    if dFrameOut: 
        print('\n Converting to dFrame ... ')
        return(cohSqDF(outFreqs, bigD))
    
    return(outFreqs, bigD)
    

def cohSqDF(freqs, bigD):
    
    '''
    Converts the bulky dictionary from the avgCohsqTShift function in to a 
    tidy pandas dataframe. Can be run separately or from within the 
    avgCohsqTShift function.
    '''
    
    a = datetime.now()
    b = a
    
    print(" -> Total %i distances" % len(bigD.keys()))
    
    for idx, dist in enumerate(bigD):
        
        if idx % 5 == 0 and idx != 0:
            print(idx, end=' ... ')
            print(str((datetime.now()-a).seconds) + ' s | ' + str((datetime.now()-b).seconds))
            a = datetime.now()
            
        for idx2, time in enumerate(bigD[dist]):
            
            dist_m = int(dist)*.254
            timelag_s = int(time)*1.288
            coh = bigD[dist][time]
            fr = {'dist_m':dist_m, 'time_s':timelag_s,
                  'freq':freqs, 'coh':coh}
            
            if idx == 0 and idx2 == 0:
                frame = pd.DataFrame(fr)
            else:
                fr = pd.DataFrame(fr)
                frame = frame.append(fr, ignore_index=True)
                
    return(frame)


def normaliseCohSpec(dataFrame, destructive=False):
    
    newFrame = dataFrame.copy(deep=True)
    
    freqs = dataFrame['freq'].unique()
    for f in freqs:
        
        Zs = dataFrame['coh'][dataFrame['freq']==f]
        Zrange = Zs.max() - Zs.min()
        Znorm = (Zs - Zs.min()) / Zrange
        
        newFrame['coh'][newFrame['freq']==f] = Znorm
        
    return(newFrame)


def cohSpecPeakExtraction(df, gaussSpan=3, gaussResolution=0.01, update=True):
    
    '''
    df NEEDS to be the dataframe output from cohSqDF / avgCohsqTShift_noPadding
    '''
    
    # important subsets
    freqs = df['freq'].unique()
    lags = df['time_s'].unique()
    distances = df['dist_m'].unique()
    
    # holding the output
    dist_m = []
    peakLag = []
    freq = []
    coh = []
    errorCount = 0
    
    print(" ** Peak extraction")
    print(" %i freqs from %.3f to %.3f" % (len(freqs), freqs[0], freqs[-1]))
    
    # routine
    for idx, f in enumerate(freqs):
        if update and idx % 50 == 0 and idx != 0: print("%i | %.3f" % (idx, f))
        for d in distances:
            #print(f,d)
            cohSlice = df['coh'][(df['dist_m']==d) & (df['freq']==f)]
                        
            peakLagIdx = np.where(cohSlice == cohSlice.max())[0]
            
            peakLagMin = int(peakLagIdx-gaussSpan)
            if peakLagMin < 0: peakLagMin = 0
            peakLagMax = int(peakLagIdx+gaussSpan+1)
            
            subLag = np.flip(lags[peakLagMin:peakLagMax])
            subCoh = np.flip(np.array(cohSlice[peakLagMin:peakLagMax]))
            #subCoh = np.multiply(subCoh, 100)
            newX = np.arange(subLag[0], subLag[-1], gaussResolution)
            
            mean = sum(subLag * subCoh) / sum(subCoh)
            sigma = np.sqrt(abs(sum(subLag * (subCoh - mean) ** 2) / sum(subCoh)))
            #print(sigma)
            
            if sigma > 1000:
                #print("SWAPPING %.2f for" % sigma, end=' ')
                subCoh = np.multiply(subCoh, 100)
                sigma = np.sqrt(abs(sum(subLag * (subCoh - mean) ** 2) / sum(subCoh)))
                #print('%.2f' % sigma)
            try:
                popt, _ = curve_fit(Gauss, subLag, subCoh, p0=[max(subCoh), mean, sigma])
                gaussFit = Gauss(newX, *popt)
                timeLagGauss = newX[np.argmax(gaussFit)]
                cohMax = np.max(gaussFit)
            except:
                timeLagGauss = lags[np.argmax(cohSlice)]
                cohMax = np.max(cohSlice)
                errorCount += 1
            
            dist_m.append(d)
            peakLag.append(timeLagGauss)
            freq.append(f)
            coh.append(cohMax)
            
    print(' done, %i errors' % errorCount)
            
    output = pd.DataFrame.from_dict({'dist_m':dist_m, 'peakLag_s':peakLag,
                                     'freq':freq, 'peakCoh':coh}) 
    return(output)
            
            
# --- Analysis Routine 1
    

def Analysis1_singleSection(inRow, data, divider=None):
    
    #global data # yolo
    global outFolder
    
    height = inRow['height_int']
    path = inRow['path']
    
    name = inRow['name']
    dense = inRow['dense']
    
    idKey = 'exp_dense' if dense else 'exp'
    ids = defaultCohIndices(idKey)
    
    if divider is None:
        # output names
        corrName = os.path.join(outFolder, name + '_CorrelationFrame_30Min.pkl')
        cohName = os.path.join(outFolder, name + '_CohSQFull_30Min.pkl')
        peakName = os.path.join(outFolder, name + '_CohSQPeaks_30Min.pkl')
    else:
        thisIter = divider[0]
        iterSize = divider[1]
        tail = str(iterSize)+'Min_'+str(thisIter).zfill(2)+'.pkl'
        
        corrName = os.path.join(outFolder, name + '_CorrelationFrame_' + tail)
        cohName = os.path.join(outFolder, name + '_CohSQFull_' + tail)
        peakName = os.path.join(outFolder, name + '_CohSQPeaks_' + tail)
        
    # subset data
    dataSub = getmeoneheight(data, height=height, side=path, mask=True, dense=dense)
    
    if not os.path.isfile(corrName):
        print("\n 1. INTEGRAL CORRELATION JUNK")
        # integral correlation stuff
        corr = fullCorrOneHeight(dataSub)
        d, t, sd = weightedAvgSegment(data=corr['timeLag_Gauss'], filter_val=corr['distance_m'],
                                  weight_val=corr['corrCoeff'], lims=True)
        correlationFrame = pd.DataFrame.from_dict({'distance':d, 'avg_timelag':t, 'sd':sd})

        correlationFrame.to_pickle(corrName)
    else:
        print("Correlation already done?")
    
    if not os.path.isfile(cohName) and not os.path.isfile(peakName):
        print("\n 2. WAVENUMBER VELOCITY JUNK")
        # wavenumber stuff
        cohSqFrame = avgCohsqTShift_noPadding(dataSub, maxTimeLag_s=120, limitDist_or_idxList=ids,
                                          nsmooth=21, nfft=None, collapse=True,
                                          dFrameOut=True, updateIter=20)
        peakFrame = cohSpecPeakExtraction(cohSqFrame)
        
        cohSqFrame.to_pickle(cohName)
        peakFrame.to_pickle(peakName)
    else: 
        print("Coherence already done?")


def DHT(x):
    '''Dense Height Translator'''
    a = {0:15, 1:14, 2:13, 3:12, 4:11, 5:10, 6:9, 7:8, 8:7, 9:6, 10:5, 11:4,
         12:3, 13:2, 14:1, 15:0}
    return(a[x])


def defaultCohIndices(key='exp'):
    if key == 'exp':
        
        ids = [1,3,5,7,9,11,13,15,17,19,
       23,27,31,35,
       40,46,52,58,64,
       70,80,90,100,
       150,200,250,300,400,500,600]
        '''
        ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 31, 35, 39,
               43, 47, 51, 55, 59, 70, 80, 90, 100, 110, 120, 130, 140,
               150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 500, 600]
        '''
    elif key == 'exp_dense':
        ids = [1,3,5,7,9,11,13,15,17,19,
       23,27,31,35,
       40,46,52,58,64,
       70,80,90,100]
    elif key == 'somethingElse':
        pass
    else: raise ValueError('invalid key')
    
    return(ids)

ids = [1,3,5,7,9,11,13,15,17,19,
       23,27,31,35,
       40,46,52,58,64,
       70,80,90,100,
       150,200,250,300,400,500,600]


def getIterables():
    
    iterables = {
    'name':['SENW_high', 'SENW_low', 'SWNE_high', 'SWNE_low', 'SENW_2', 'SENW_3', 'SENW_4', 'SENW_5', 'SENW_6',
             'SENW_7', 'SENW_8', 'SENW_9', 'SENW_10', 'SENW_11', 'SENW_12', 'SENW_13', 'SENW_14', 'SENW_15',
             'SWNE_2', 'SWNE_3', 'SWNE_4', 'SWNE_5', 'SWNE_6', 'SWNE_7', 'SWNE_8', 'SWNE_9', 'SWNE_10',
             'SWNE_11', 'SWNE_12', 'SWNE_13', 'SWNE_14', 'SWNE_15'],
    'height_int':[0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'dense':[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'path':['SENW', 'SENW', 'SWNE', 'SWNE', 'SENW', 'SENW', 'SENW', 'SENW', 
            'SENW', 'SENW', 'SENW', 'SENW', 'SENW', 'SENW', 'SENW', 'SENW', 
            'SENW', 'SENW', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 
            'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE', 'SWNE']}
    
    return(pd.DataFrame.from_dict(iterables))


def chopDTSDict(data, divider):
    
    outData = {} # dic for other dics
    nSteps = len(data['time'])
    segmentSize = nSteps // divider
    
    print(" -> Chopping 30-min in %i" % divider)
    print(" -> %i total indices, %i per division" % (nSteps, segmentSize))
    
    # This is the line that is causing an extra key to appear in the divided
    # dictionary. When I do this now the list comprehension creates the 
    # necessary indices, and then when the final one is appended it creates
    # a third key (when I only want, e.g., 2) that is either empty or carries
    # the remainder of keys that don't fit in (like... 1 max maybe?) Looking
    # at this now (Jan. 2021) I don't know why it's written this way but
    # I remember being frustrated by something before so I'm going to leave
    # it the way it is. Because dics are ordered now we can safely ignore the
    # final one and assume it has barely any entries (an updated log could
    # confirm this by checking the length of the segment)
    ids = [x for x in range(0, nSteps, segmentSize)]
    ids.append(nSteps-1)
    
    for n, i in enumerate(range(len(ids)-1)): 
        sI = ids[i]
        eI = ids[i+1]
        
        d = {}
        for key in data:
            if isinstance(data[key], np.ndarray):
                d[key] = data[key][sI:eI,:,:]
            else:
                d[key] = data[key][sI:eI]
        
        outData[str(n).zfill(3)] = d
        
    return(outData)
    
#%% Long Routine

if __name__ == '__main__':
    
    # determine OS -> keep script consistent between platforms
    OS = sys.platform

    start = 0
    step = 1
    limit = False
    divider = 1

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 's:i:hld:')
        #print(opts)
        for opt, arg in opts:
            if opt == '-s':
                start = int(arg)
                print("Overriding folder start idx with %i" % start)
            elif opt == '-i':
                step = int(arg)
                print("Overriding default step with %i" % step)
            elif opt == '-l':
                limit = True
            elif opt == '-d':
                d = int(arg)
                if d >= 1:
                    divider = d
                else:
                    print(" ** Divider can't be less than 1, reverting to default")
            elif opt == '-h':
                print("Possible arguments:")
                print(" -> -s (int) sets the start index in sorted list of subfolders (default=0)")
                print(" -> -i (int) sets the step, i.e. -i 2 runs every second folder (default=1)")
                print(" -> -l stands for limit: it will only run once (set with -s)")
                print(" -> -d (int) is the divider: splits the 30-min chunks in to d equal sizes (default=1)")
                print(" * NB: the base folder needs to be changed inside the .py file")
                sys.exit()
    except:
        print("Something janky happened")
        sys.exit(1)
        
        
    # gets all the possible subFolders
    if OS == 'darwin':
        baseFolder = '/Users/rainerhilland/DTS_Fork'
    elif OS == 'linux':
        baseFolder = '/home/rainer/data/'
    subFolders = glob.glob(os.path.join(baseFolder, '*'))
    subFolders = [x for x in subFolders if os.path.isdir(x)]
    subFolders.sort()
    
    iterables = getIterables()
    
    for subFolder in subFolders[start:][::step]: # only do a portion
    
        print('* ---------------------------------------------------------- *')
        print(subFolder, end='\n\n\n')
        
        fileName = os.path.join(subFolder, 'mappedDTSDicts_InterpAndMean.h5')
    
        if os.path.exists(os.path.join(subFolder, 'note.txt')):
            print("Skipping %s - Data Missing" % subFolder)
            continue
        
        if ((x:= len(glob.glob(os.path.join(subFolder, 'analysis', '*.pkl')))) > 30):
            print("%i files already here, probably done" % x)
    
        if not os.path.exists(fileName):
            print("Data missing in %s??" % subFolder)
            continue
            
        data = readMappedArray(fileName)
        
        if divider != 1:
            dataDic = chopDTSDict(data, divider)
        outFolder = os.path.join(subFolder, 'analysis')
        if not os.path.isdir(outFolder): os.mkdir(outFolder)
        
        for idx, row in iterables.iterrows():
            print("\n\n * -- NEW SECTON -- *")
            print(subFolder)
            print(datetime.now())
            print(idx,row)
            if divider == 1:
                Analysis1_singleSection(row, data)
            else:
                divSize = int(30 / divider)
                for i, key in enumerate(dataDic):
                    if len(dataDic[key]['time'] <= 1):
                        print("Skipping key %s, it's empty..." % key)
                        continue
                    i += 1
                    print(" -> Part %i of %i (len: %i)" % 
                          (i, divider, len(dataDic[key]['time'])))
                    Analysis1_singleSection(row, dataDic[key], (i, divSize))

        if limit: sys.exit()
    
