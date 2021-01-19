#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:47 2020

@author: rainer
"""

import sys, getopt
import os, glob

# checker for complete folders

if __name__ == '__main__':
    
    dataFolders = '/home/rainer/data'
    nFilesTarget = 96
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:n:")
        for opt, arg in opts:
            if opt == '-d':
                dataFolders = arg
            elif opt == '-n':
                nFilesTarget = int(arg)
    except:
        print("wtf")
        
    print("Scanning for complete/incomplete folders in")
    print("%s" % dataFolders)
    
    subFolders = glob.glob(os.path.join(dataFolders, '*'))
    subFolders.sort()
    
    print("Found %i subFolders to check" % len(subFolders))
    if len(subFolders) == 0:
        print("Exiting")
        sys.exit(1)
        
    goods = 0
    underway = 0
    fishy = 0
    total = len(subFolders)
        
    for sub in subFolders:
        if os.path.isfile(os.path.join(sub, 'note.txt')):
            print("%s | no data to analyse -> OK" % sub)
            goods += 1
            
        else:
            checkFolder = os.path.join(sub, 'analysis')
            nFiles = len(glob.glob(os.path.join(checkFolder, '*')))
            if nFiles == nFilesTarget:
                print("%s | folder completed (%i files) -> OK" % (sub, nFiles))
                goods += 1
            elif nFiles > 0 and nFiles < nFilesTarget:
                print("%s | folder in progress (%i files)" % (sub, nFiles))
                underway += 1
            elif nFiles == 0:
                print("%s | folder not started" % sub)
            else:
                print("%s | something's fishy (%i files)" % (sub, nFiles))
                fishy += 1
            
    print(" -- CHECK FINISHED")
    print(" -> %i total folders" % total)
    print(" -> %i/%i finished" % (goods, total))
    print(" -> %i/%i in progress" % (underway, total))
    print(" -> %i/%i fishy" % (fishy, total))
    print(" -> %i/%i waiting to be done" % (total-(goods+underway+fishy), total))
    sys.exit()
                
                
        
