import sys

sys.path.append('/Users/RainerHilland/Documents/UWO CourseWork/PhD/Scripts/NamTEX/DroneThermography') # fucking 3.8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, glob
import videoWriter as vW
#from celluloid import Camera
from scipy.signal.windows import gaussian
from scipy.ndimage import filters


class DTS:

    '''
    This class will read/build/format a DTS file.
    Pass a filename to read, pipeline is disjointed at the moment.
    
    TODO: Function descriptions!
    
    Remember you can do temporal AND spatial averaging; make use of that.
    '''

    def __init__(self, fname, justMeta=False):
        
        self.filename = fname
        self.buildMetadata()
        if not justMeta:
            self.readData()


    def buildMetadata(self):
        
        ''' This collects the header data from the txt files. '''
        
        self.metadata = pd.read_csv(self.filename, nrows=10, header=None)
        self.sampleStart = self.metadata[0][0]
        self.sampleEnd = self.metadata[0][1]
        self.sampleDuration = self.metadata[0][2]
        self.startDist = float(self.metadata[0][3])
        self.endDist = float(self.metadata[0][4])
        self.sampleInterval = float(self.metadata[0][5])
        self.channel = int(self.metadata[0][6])
        self.internalTemp = float(self.metadata[0][7])
        self.p1Temp = float(self.metadata[0][8])
        self.p2Temp = float(self.metadata[0][9])


    def readData(self):

        colnames = ["LAF","ST","AST","T"]
        self.data = pd.read_csv(self.filename, names=colnames, skiprows=10,
                                header=None)


    def plot(self):

        plt.plot(self.data['LAF'], self.data['T'])
        l = "t: %s P1: %.2f P2: %.2f PI: %.2f" % (self.sampleEnd, self.p1Temp, self.p2Temp, self.internalTemp)
        plt.title(l)
        plt.show()


    def forceData(self, newData):
        self.data = newData


    def help(self, start, end=None):
        
        if end is None:
            end = start + 10

        nearestStart = min(self.data['LAF'], key=lambda x:abs(x-start))
        nearestEnd = min(self.data['LAF'], key=lambda x:abs(x-end))

        startIndex = np.where(self.data['LAF'] == nearestStart)[0][0]
        endIndex = np.where(self.data['LAF'] == nearestEnd)[0][0]

        print("%i to %i, indices %i to %i" % (start, end, startIndex, endIndex))
        print(self.data[['LAF','T']][startIndex:(endIndex+1)])


    def labelPlot(self, start, end):

        nS = min(self.data['LAF'], key=lambda x:abs(x-start))
        nE = min(self.data['LAF'], key=lambda x:abs(x-end))
        sI = np.where(self.data['LAF'] == nS)[0][0]
        eI = np.where(self.data['LAF'] == nE)[0][0]

        plt.plot(self.data['LAF'], self.data['T'])
        for i, num in enumerate(self.data['LAF'][sI:eI]):
            if i % 2 == 1: continue
            plt.annotate(round(num, 3), (num, self.data['T'][sI+i]), rotation=90)

        plt.show()


    def denseMap(self, map=102, size=106):

        """ map can point to an already loaded file or to a .csv that needs to be loaded! """

        # this is state of the art as of 28.07.2020
        mapFolder = '/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/Data/DTS/DTSMaps/'
        
        if map == 102:
            mapArray = pd.read_csv('/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/Data/DTS/DTSMaps/DTSMap_V01_T02.csv')
        elif map == 202:
            mapArray = pd.read_csv('/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/Data/DTS/DTSMaps/DTSMap_V02_T02.csv')
        elif map == 302:
            mapArray = pd.read_csv('/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/Data/DTS/DTSMaps/DTSMap_V03_T02.csv')
            

        NESW = np.empty((16,size)) # assuming we don't exceed 106!!!
        SENW = np.empty((16,size))

        for idx in range(2,18):
            sI = mapArray['idxStart'][idx]
            sE = mapArray['idxEnd'][idx]+1 # forgot this one you idiot!

            temp = np.empty(size)
            thisChunk = self.data['T'][sI:sE]

            if mapArray['reverse'][idx]:
                thisChunk = np.flip(thisChunk)

            temp[0:temp.shape[0]] = np.nan
            temp[0:thisChunk.shape[0]] = thisChunk

            NESW[idx-2,:] = temp

        for idx in range(18,34):
            sI = mapArray['idxStart'][idx]
            sE = mapArray['idxEnd'][idx]+1

            temp = np.empty(size)
            thisChunk = self.data['T'][sI:sE]

            if mapArray['reverse'][idx]:
                thisChunk = np.flip(thisChunk)

            temp[0:temp.shape[0]] = np.nan
            temp[0:thisChunk.shape[0]] = thisChunk

            SENW[idx-18,:] = temp

        self.dense = [NESW, SENW]
        self.mapped = 1


    def densePlot(self, save=False, name=None):

        if not self.mapped: raise ValueError("Map data first")

        high = np.nanmax(self.dense)
        low = np.nanmin(self.dense)

        fig, ax = plt.subplots(2,1,figsize=(12,6))

        im=ax[0].imshow(self.dense[0], vmax=high, vmin=low)
        ax[0].set_title("NE =-> SW")

        ax[1].imshow(self.dense[1], vmax=high, vmin=low)
        ax[1].set_title("SE =-> NW")

        plt.colorbar(im, ax=ax[1], orientation='horizontal')
        plt.suptitle(self.sampleStart)

        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()
            
            
    def longMap(self, day):
        
        '''
        day needs to be a string in form DDMM, e.g. August 8 map is '0803'
        
        indexing direction: (not yet implemented)
            1 = NW
            2 = SE
            3 = SW
            4 = NE
            
        indexing height:
            0 = high
            1 = low
            
        0-index on x axis is towards the south, increase towards the right.
        i.e. on directions 2, 3 the left-side of a plot is the 15th stake, right side is the centre
            on directions 1, 4 the left-side of a plot is the centre, the right side is stake 15
            
        --> consider adding version control to this as well! Not just defining the days but also
            using an interpolation method like RV suggested. Could perhaps then be automated?
            
        '''

        # TODO: this needs to be converted to handle the datastore format from the dtscalibration routine
        # --> Need to probably re-do this entire piece of code :(
        # --> wrap dtscalibration around a custom routine to handle the DTS in ~30 minute chunks?
        
        # would be nicer to not hard code the map folder, but w/e for now
        mapFolder = '/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/Data/DTS/DTSMaps'
        longMap = glob.glob(os.path.join(mapFolder, '*_Long*'+day+'.csv'))[0] 
        mapArray = pd.read_csv(longMap)
        
        # ugly! and way too hardcoded!
        
        # =-> NORTH-WEST
        startIdx = mapArray['idx'][0]
        endIdx = mapArray['idx'][1]
        NWLow = self.data['T'][startIdx:endIdx+1]
        NWLow = pointDeleter(mapArray, NWLow, 2, 15, False)
        
        startIdx = mapArray['idx'][16]
        endIdx = mapArray['idx'][17]
        NWHigh = self.data['T'][startIdx:endIdx+1]
        NWHigh = pointDeleter(mapArray, NWHigh, 18, 31, True)
        
        NW = np.empty((2,NWHigh.shape[0]))
        NW[0,:] = NWHigh
        NW[1,:] = NWLow
        
        # =-> SOUTH-EAST
        startIdx = mapArray['idx'][32]
        endIdx = mapArray['idx'][33]
        SELow = self.data['T'][startIdx:endIdx+1]
        SELow = pointDeleter(mapArray, SELow, 34, 47, True)
        
        startIdx = mapArray['idx'][48]
        endIdx = mapArray['idx'][49]
        SEHigh = self.data['T'][startIdx:endIdx+1]
        SEHigh = pointDeleter(mapArray, SEHigh, 50, 63, False)
        
        SE = np.empty((2,SEHigh.shape[0]))
        SE[0,:] = SEHigh
        SE[1,:] = SELow
    
        # =-> SOUTH-WEST
        startIdx = mapArray['idx'][64]
        endIdx = mapArray['idx'][65]
        SWLow = self.data['T'][startIdx:endIdx+1]
        SWLow = pointDeleter(mapArray, SWLow, 66, 79, True)
        
        startIdx = mapArray['idx'][80]
        endIdx = mapArray['idx'][81]
        SWHigh = self.data['T'][startIdx:endIdx+1]
        SWHigh = pointDeleter(mapArray, SWHigh, 82, 95, False)
        
        # fixing cable break (TODO: improve this!)
        difference = SWHigh.shape[0] - SWLow.shape[0]
        #SWLow = np.flip(SWLow)
        a = SWLow[:300]
        b = SWLow[300:]
        for i in range(difference):
            a = np.append(a, np.nan) # or pop in last obs?
        SWLow = np.append(a,b)
        SWLow = np.flip(SWLow)
        
        SW = np.empty((2, SWHigh.shape[0]))
        SW[0,:] = SWHigh
        SW[1,:] = SWLow
        
        # =-> NORTH-EAST
        startIdx = mapArray['idx'][96]
        endIdx = mapArray['idx'][97]
        NELow = self.data['T'][startIdx:endIdx+1]
        NELow = pointDeleter(mapArray, NELow, 98, 111, False)
        
        startIdx = mapArray['idx'][112]
        endIdx = mapArray['idx'][113]
        NEHigh = self.data['T'][startIdx:endIdx+1]
        NEHigh = pointDeleter(mapArray, NEHigh, 114, 127, True)
        
        NE = np.empty((2, NEHigh.shape[0]))
        NE[0,:] = NEHigh
        NE[1,:] = NELow
        
        self.NW = NW
        self.SE = SE
        self.SW = SW
        self.NE = NE
        self.longMapped = 1
        
        self.longMax = max(np.nanmax(self.NW), np.nanmax(self.SE), 
                           np.nanmax(self.SW), np.nanmax(self.NE))
        self.longMin = min(np.nanmin(self.NW), np.nanmin(self.SE),
                           np.nanmin(self.SW), np.nanmin(self.SE))
        

    def longPlot(self, save=False, name=None):
        
        if not self.longMapped:
            raise ValueError('need to map dense first')
        
        fig,ax = plt.subplots(4, 1, figsize=(18, 16))
        
        im=ax[0].imshow(self.NW, vmax=self.longMax, vmin=self.longMin, 
                        aspect='auto', interpolation='none', cmap='jet')
        ax[0].set_title("Centre =-> NW")
        
        ax[1].imshow(self.NE, vmax=self.longMax, vmin=self.longMin, 
                     aspect='auto', interpolation='none', cmap='jet')
        ax[1].set_title("Centre =-> NE")
        
        ax[2].imshow(self.SW, vmax=self.longMax, vmin=self.longMin, 
                     aspect='auto', interpolation='none', cmap='jet')
        ax[2].set_title("SW =-> Centre")
        
        ax[3].imshow(self.SE, vmax=self.longMax, vmin=self.longMin, 
                     aspect='auto', interpolation='none', cmap='jet')
        ax[3].set_title("SE =-> Centre")
        
        plt.colorbar(im, ax=ax[3], orientation='horizontal')
        plt.suptitle(self.metadata[0][1])
        
        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()

    
    def __repr__(self):
        return "DTS file at %s" % self.sampleEnd


    def __str__(self):
        return "DTS File:\nTime: %s UTC\nP1: %.2f\nP2: %.2f" % (self.sampleEnd, self.p1Temp, self.p2Temp)
        


def pointDeleter(mapArray, fibreSegment, start, end, reverse):
    
    '''handles the point deletion for the long-segment mapping, just
    cleaning some repetitive code'''
    
    for idx in range(start, end+1):
        if mapArray['type'][idx] == 1:
            toDel = mapArray['idx'][idx] # this is the peak to be deleted
            newPt = (fibreSegment.loc[toDel-1] + fibreSegment.loc[toDel+1]) / 2 # avg adjacent points
            
            fibreSegment.loc[toDel-1] = newPt # assign new point
            fibreSegment = fibreSegment.drop([toDel,toDel+1]) # delete points
            
        elif mapArray['type'][idx] == 2:
            toDel = mapArray['idx'][idx]
            fibreSegment = fibreSegment.drop([toDel, toDel+1]) # delete two points
            
    if reverse: fibreSegment = np.flip(fibreSegment)
            
    return(fibreSegment)


def smooth(files, stat='mean', verbose=True):

    nFiles = len(files)
    if verbose: print("Smoothing {0} files".format(nFiles))

    colnames = ["LAF","ST","AST","T"]

    laf = pd.read_csv(files[0], names=colnames, skiprows=10, header=None)["LAF"]

    avgT = np.empty((nFiles, len(laf)))
    ST = np.empty((nFiles, len(laf)))
    AST = np.empty((nFiles, len(laf)))

    for idx, f in enumerate(files):
        data = pd.read_csv(f, names=colnames, skiprows=10, header=None)
        temps = data["T"]
        stokes = data["ST"] # this has to be the dumbest way to do this
        astokes = data["AST"]
        try:
            avgT[idx,:] = temps
            ST[idx,:] = stokes
            AST[idx,:] = astokes
        except:
            print("READ ERROR AT %i %s" % (idx, f))
            try:
                if idx != 0:
                    avgT[idx,:] = avgT[idx-1,:]
                    ST[idx,:] = ST[idx-1,:]
                    AST[idx,:] = AST[idx-1,:]
            except:
                continue
            continue

    if stat == 'mean':
        avgT = np.mean(avgT, axis=0)
        ST = np.mean(ST, axis=0)
        AST = np.mean(AST, axis=0)
    elif stat == 'sd':
        avgT = np.std(avgT, axis=0)
        #avgT = avgT/np.mean(avgT)
        ST = np.std(ST, axis=0)
        AST = np.std(AST, axis=0)

    data = {'LAF':laf, 'ST':ST, 'AST':AST, 'T':avgT}
    df = pd.DataFrame(data)

    smoothedFile = DTS(files[nFiles//2]) # metadata comes from ~ the middle
    smoothedFile.forceData(df)

    return(smoothedFile)


def plotTwo(A, B):

    plt.plot(A.data['LAF'], A.data['T'])
    plt.plot(B.data['LAF'], B.data['T'])
    plt.show()


def demo(files, n=1):

    plotFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/TEMP - DELETE"
    writeFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/TestingRawData/DTS_JULY"
    cleanUp(plotFolder, '*.png')
    for f in files:

        name = os.path.splitext(os.path.basename(f))[0]
        newName = os.path.join(plotFolder, name + '.png')
        
        thisFile = DTS(f)
        thisFile.map1()
        thisFile.plotMapped1(save=True, name=newName)

    vName = str(n)+'_demo.avi'
    vW.writeVideo(plotFolder, writeFolder, videoName=vName, ext='*.png', frameRate=20)


def cleanUp(folder, of):
    toClean = glob.glob(os.path.join(folder, of))
    dummy = [os.remove(f) for f in toClean]


def flexiPlot():
    pass


def denseAnimPlot(files, n, smoothFiles=False, by=None, ver=202, filter1D=False, goodAxes=True):

    plotFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/TEMP - DELETE"
    writeFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/Data/TestingRawData/2020.08 DTS Mapping"
    cleanUp(plotFolder, '*.png')

    #fyi we're using an assumed 2x16x106 dense shape here
    nFiles = len(files)
    timing=[]

    # initialise empty full array
    # indexing is (frame,direction,transect,distance),
    # i.e. (nFrames,2,16,106)
    if smoothFiles:
        fullArray = np.empty((nFiles//by, 2, 16, 106))
    else:
        fullArray = np.empty((nFiles, 2, 16, 106))

    print("Expect %i files: " % fullArray.shape[0])

    # populate the array
    if smoothFiles: # only temporal smoothing at the mo
        for counter, idx in enumerate(range(len(files))[::by]): # another way is to do rolling smoothing?
            if counter % 25 == 0: print('%i ...' % idx, end='')
            if counter == fullArray.shape[0]: continue
            
            theseBoys = files[idx:idx+by]
            data = smooth(theseBoys, verbose=False)
            data.denseMap(ver)
            
             # this gives it the time of ~the middle (see smooth function)
            timing.append(data.sampleStart)
            
            fullArray[counter,0,:,:] = data.dense[0]
            fullArray[counter,1,:,:] = data.dense[1]
            
    else:
        for idx, file in enumerate(files):

            if idx % 25 == 0: print("%i ..." % idx, end='')

            data = DTS(file)
            data.denseMap(ver)

            timing.append(data.sampleStart)

            fullArray[idx,0,:,:] = data.dense[0]
            fullArray[idx,1,:,:] = data.dense[1]
            
            

    print('done')
    
    if filter1D: 
        print("Filtering ... ", end='')
        fullArray = DTSSmoothTime(fullArray)
        print('done')

    # vmax/vmin to keep colour bar consistent
    high = np.nanmax(fullArray)
    low = np.nanmin(fullArray)

    print('Plotting ... ')
    # plot them
    for idx in range(fullArray.shape[0]):

        if idx % 25 == 0: print("%i ..." % idx, end='')

        outName = os.path.join(plotFolder, timing[idx] + '.png')

        if goodAxes:
            fig,ax=plt.subplots(2,1,figsize=(12,4))
        else:
            fig,ax = plt.subplots(2,1,figsize=(12,6))

        im=ax[0].imshow(fullArray[idx,0,:,:], vmax=high, vmin=low, cmap='jet', interpolation='none')
        ax[0].set_title("NE =-> SW")

        ax[1].imshow(fullArray[idx,1,:,:], vmax=high, vmin=low, cmap='jet', interpolation='none')
        ax[1].set_title("SE =-> NW")

        plt.colorbar(im, ax=ax[1], orientation='horizontal')
        plt.suptitle(timing[idx])
        
        if goodAxes:
            ax[0].set_aspect(0.66)
            ax[1].set_aspect(0.66)
            

        plt.savefig(outName)
        plt.close(fig)

    print('done')

    # make the video
    vW.writeVideo(plotFolder, writeFolder, videoName=str(n)+'_demo.avi', ext='*.png', frameRate=16)


def DTSArray(files):
    
    # only dense sections for now: (frames,dir,height,dist); (n,2,16,106)
    
    nFiles = len(files)
    array = np.empty((nFiles,2,16,106))
    
    print("Expect %i files:" % nFiles)
    for idx, file in enumerate(files):
        
        if idx % 25 == 0: print('%i ... ' % idx, end='')
        
        data = DTS(file)
        data.map1()
        
        array[idx,0,:,:] = data.dense[0]
        array[idx,1,:,:] = data.dense[1]
        
    print('done')
    return(array)


def DTSSmoothTime(array, sd=2):
    
    # assumes an (n,2,16,106) array
    # basically a port of decompFunctions.filterArray1D()
    # non-destructive
    
    window = gaussian(30, sd)
    
    if len(array.shape) == 4:
        smoothArray = np.empty(array.shape)

    
        for row in range(16):
            for col in range(106):
            
                smoothArray[:,0,row,col] = filters.convolve1d(array[:,0,row,col],
                                                          window/window.sum())
                smoothArray[:,1,row,col] = filters.convolve1d(array[:,1,row,col],
                                                              window/window.sum())
    
    elif len(array.shape) == 3:
        smoothArray = np.empty(array.shape)
        
        for row in range(array.shape[1]):
            for col in range(array.shape[2]):
                smoothArray[:,row,col] = filters.convolve1d(array[:,row,col],
                                                          window/window.sum())
        
    return(smoothArray)
    

def NWTest(dts):
    
    """
    
    *** DEPRECATED: THIS IS/WAS ONLY FOR TESTING! ***
    
    This approach basically works I think. Take the start/end indices of the whole
    transect and then just pick out the stake points using the 1- or 2-peak method. Can
    probably be improved, and I may need to come up w/ multiple maps if the cable shifts
    too much.
    
    17.07 TODO: write the rest of the map(s) and tidy up this function (lots of repetitive code)
    18.07: DO NOT USE, implemented as the DTS.longMap method
    """
    
    
    mapArray = pd.read_csv('/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/FieldNotes/DTSMap_Long_V01_0803.csv')
    
    # pretty hard-coded
    startIdx = mapArray['idx'][0]
    endIdx = mapArray['idx'][1]
    
    NWLow = dts.data['T'][startIdx:endIdx+1]
    
    for idx in range(2,16):
        
        if mapArray['type'][idx] == 1:
            
            toDel = mapArray['idx'][idx]
            newPt = (NWLow.loc[toDel-1] + NWLow.loc[toDel+1]) / 2
            
            NWLow.loc[toDel-1] = newPt
            NWLow = NWLow.drop([toDel,toDel+1])
            
        elif mapArray['type'][idx] == 2:
            
            toDel = mapArray['idx'][idx]
            NWLow = NWLow.drop([toDel,toDel+1])
            
    startIdx = mapArray['idx'][16]
    endIdx = mapArray['idx'][17]
    
    NWHigh = dts.data['T'][startIdx:endIdx+1]
    
    for idx in range(18, 32):
        
        if mapArray['type'][idx] == 1:
            
            toDel = mapArray['idx'][idx]
            newPt = (NWHigh.loc[toDel-1] + NWHigh.loc[toDel+1]) / 2
            
            NWHigh.loc[toDel-1] = newPt
            NWHigh = NWHigh.drop([toDel,toDel+1])
            
        elif mapArray['type'][idx] == 2:
            
            toDel = mapArray['idx'][idx]
            NWHigh = NWHigh.drop([toDel,toDel+1])
            
            
    # 0 is high, 1 is low so that imshow interprets correctly
    # left edge is stake 15, right edge is centre
    NW = np.empty((2,NWHigh.shape[0]))
    NW[0,:] = NWHigh
    NW[1,:] = np.flip(NWLow)
    
    return(NW)


def NWPlotTest(files, n, smooth):
    
    plotFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/NamTEX/TEMP - DELETE"
    writeFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/TestingRawData/DTS_JULY"
    cleanUp(plotFolder, '*.png')
    
    #fyi we're using an assumed 2x16x106 dense shape here
    nFiles = len(files)
    timing=[]

    # initialise empty full array    
    fullArray = np.empty((nFiles, 2, 593))

    print("Expect %i files: " % fullArray.shape[0])

    # populate the array
    
    for idx, file in enumerate(files):

        if idx % 25 == 0: print("%i ..." % idx, end='')

        data = DTS(file)
        transect = NWTest(data)

        timing.append(data.sampleStart)

        fullArray[idx,:,:] = transect
        
        if idx == 0:
            high = np.nanmax(transect)
            low = np.nanmin(transect)
        else:
            nhigh = np.nanmax(transect)
            nlow = np.nanmin(transect)
            if nhigh > high: high = nhigh
            if nlow < low: low = nlow
        
            

    print('done')

    if smooth:
        fullArray = DTSSmoothTime(fullArray)


    print('Plotting ... ')
    # plot them
    for idx in range(fullArray.shape[0]):

        if idx % 25 == 0: print("%i ..." % idx, end='')

        outName = os.path.join(plotFolder, timing[idx] + '.png')

        fig,ax = plt.subplots(figsize=(18,4))

        im=ax.imshow(fullArray[idx,:,:], vmax=high, vmin=low, aspect='auto', interpolation='none',
                     cmap='jet')

        plt.colorbar(im, ax=ax, orientation='horizontal')
        plt.suptitle(timing[idx])

        plt.savefig(outName)
        plt.close(fig)
        
    vW.writeVideo(plotFolder, writeFolder, videoName=str(n)+'_demo.avi', ext='*.png', frameRate=16)


    print('done')


if __name__ == '__main__':

    folder = '/Volumes/NamTEX_DATA/NamTEX/DTS/10.03/Formatted'
    files = glob.glob(os.path.join(folder, '*.txt'))
    files.sort()
    testFile = DTS(files[5])
    
    '''
    m = 10
    
    for idx in range(len(files))[::4000]:
        
        n = str(m)+'_fullWalk12'
        m += 1
        
        denseAnimPlot(files[idx:idx+4000], n, smoothFiles=True, by=4)
    '''
    '''
    writeFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/TestingRawData/DTS_JULY"
    a=1
    for idx in range(len(files))[::2000]:
        
        sName = str(a) + '_bigAvgs.png'
        sName = os.path.join(writeFolder, sName)
        a += 1
        j = smooth(files[idx:idx+2000])
        j.map1()
        j.plotMapped1(save=True, name=sName)
    '''
    '''
    a = DTSArray(files[40000:42000])

    plt.plot(a[:,0,5,5])
    plt.plot(b[:,0,5,5])
    '''
    '''
    writeFolder = "/Users/RainerHilland/Documents/UWO CourseWork/PhD/TestingRawData/DTS_JULY"
    for n in range(5000,10000,1000):
        
        name = os.path.join(writeFolder,str(n)+'_LONGTEST.png')
        p = smooth(files[n:n+1000])
        p.longMap('0803')
        p.longPlot(save=True, name=name)
    '''
    '''
    # Sep. 2 -> animations for NamTEX Symp
    # Dense animation plots for 12.03 14:00:00 to 14:30:00, both raw and time-smoothed
    startI = 38149
    endI = 39501
    
    #name = '40_halfHour_Raw'
    #denseAnimPlot(files[startI:endI], name, goodAxes=True)
    
    #name = '41_halfHour_4smooth'
    #denseAnimPlot(files[startI:endI], name, smoothFiles=True, by=3, goodAxes=True)
    
    #name = '43_halfHour_2smooth_filt_mp4'
    #denseAnimPlot(files[startI:endI], name, smoothFiles=True, by=2, filter1D=True, goodAxes=False)
    
    name = '45_halfHour_2smooth_filt_NIGHT1'
    denseAnimPlot(files[12252:13950], name, smoothFiles=True, by=2, filter1D=True, goodAxes=False)
    '''
